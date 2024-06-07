//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file one_particle.cpp
//  \brief tests one particle.

// C++ standard libraries
#include <cmath>  // round()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../particles/particles.hpp"

#if PARTICLES != DUST_PAR
#error "This problem generator requires dust particles"
#endif

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Get the (uniform) velocity of the gas.
  Real ux0, uy0, uz0;
  ux0 = pin->GetOrAddReal("problem", "ux0", 0.0);
  uy0 = pin->GetOrAddReal("problem", "uy0", 0.0);
  uz0 = pin->GetOrAddReal("problem", "uz0", 0.0);

  // Set a uniform, steady gas.
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->u(IDN,k,j,i) = 1.0;
        phydro->u(IM1,k,j,i) = ux0;
        phydro->u(IM2,k,j,i) = uy0;
        phydro->u(IM3,k,j,i) = uz0;
        phydro->u(IEN,k,j,i) = 1.0 / (peos->GetGamma() - 1.0);
      }
    }
  }

  // Get the dust-to-gas ratio and the velocity of the particles.
  Real dtog, vpx0, vpy0, vpz0;
  dtog = pin->GetOrAddReal("problem", "dtog", 1.0);
  vpx0 = pin->GetOrAddReal("problem", "vpx0", 0.0);
  vpy0 = pin->GetOrAddReal("problem", "vpy0", 0.0);
  vpz0 = pin->GetOrAddReal("problem", "vpz0", 0.0);

  // Find the total number of particles in each direction.
  if (PARTICLES == DUST_PAR) {
    RegionSize& mesh_size = pmy_mesh->mesh_size;
    int npx1 = (block_size.nx1 > 1)
                   ? pin->GetOrAddInteger("problem", "npx1", mesh_size.nx1)
                   : 1,
        npx2 = (block_size.nx2 > 1)
                   ? pin->GetOrAddInteger("problem", "npx2", mesh_size.nx2)
                   : 1,
        npx3 = (block_size.nx3 > 1)
                   ? pin->GetOrAddInteger("problem", "npx3", mesh_size.nx3)
                   : 1;

    // Find the mass of each particle and the distance between adjacent
    // particles.
    Real vol = (mesh_size.x1max - mesh_size.x1min) *
               (mesh_size.x2max - mesh_size.x2min) *
               (mesh_size.x3max - mesh_size.x3min);
    Real dx1 = (mesh_size.x1max - mesh_size.x1min) / npx1,
         dx2 = (mesh_size.x2max - mesh_size.x2min) / npx2,
         dx3 = (mesh_size.x3max - mesh_size.x3min) / npx3;
    DustParticles::SetOneParticleMass(dtog * vol / (npx1 * npx2 * npx3));

    // Determine number of particles in the block.
    int npx1_loc = static_cast<int>(
            std::round((block_size.x1max - block_size.x1min) / dx1)),
        npx2_loc = static_cast<int>(
            std::round((block_size.x2max - block_size.x2min) / dx2)),
        npx3_loc = static_cast<int>(
            std::round((block_size.x3max - block_size.x3min) / dx3));
    int npar = ppar->npar = npx1_loc * npx2_loc * npx3_loc;
    if (npar > ppar->nparmax) ppar->UpdateCapacity(npar);

    // Assign the particles.
    int ipar = 0;
    for (int k = 0; k < npx3_loc; ++k) {
      Real zp1 = block_size.x3min + (k + 0.5) * dx3;
      for (int j = 0; j < npx2_loc; ++j) {
        Real yp1 = block_size.x2min + (j + 0.5) * dx2;
        for (int i = 0; i < npx1_loc; ++i) {
          Real xp1 = block_size.x1min + (i + 0.5) * dx1;
          ppar->xp(ipar) = xp1;
          ppar->yp(ipar) = yp1;
          ppar->zp(ipar) = zp1;
          ppar->vpx(ipar) = vpx0;
          ppar->vpy(ipar) = vpy0;
          ppar->vpz(ipar) = vpz0;
          ++ipar;
        }
      }
    }
  }
}
