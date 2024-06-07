//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file one_particle.cpp
//  \brief tests one particle.

// C++ headers
#include <cmath>      // abs()
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../field/field.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../particles/particles.hpp"

#if PARTICLES == 0
#error "This problem generator requires particles"
#endif

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//======================================================================================

namespace {
constexpr Real uniform_density = 1.0;
} // namespace

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Get the (uniform) velocity of the gas.
  Real ux0, uy0, uz0, bx0(0), by0(0), bz0(0);
  Real B0, pres;
  int Bdir;
  ux0 = pin->GetOrAddReal("problem", "ux0", 0.0);
  uy0 = pin->GetOrAddReal("problem", "uy0", 0.0);
  uz0 = pin->GetOrAddReal("problem", "uz0", 0.0);
  if (NON_BAROTROPIC_EOS) {
    pres = pin->GetReal("problem", "pres");
  } else {
    Real iso_cs(peos->GetIsoSoundSpeed());
    pres = uniform_density * SQR(iso_cs);
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    Real beta(pin->GetReal("problem", "beta"));
    B0 = std::sqrt(static_cast<Real>(2.0 * pres / beta));
    Real ang_2 = pin->GetOrAddReal("problem", "ang_2", -999.9);
    Real ang_3 = pin->GetOrAddReal("problem", "ang_3", -999.9);
    Real sin_a2, cos_a2, sin_a3, cos_a3;
    Real x1size = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
    Real x2size = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
    Real x3size = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;
    // User should never input -999.9 in angles
    if (ang_3 == -999.9) ang_3 = std::atan(x1size / x2size);
    sin_a3 = std::sin(ang_3);
    cos_a3 = std::cos(ang_3);
    if (ang_2 == -999.9)
      ang_2 = std::atan(0.5 * (x1size * cos_a3 + x2size * sin_a3) / x3size);
    sin_a2 = std::sin(ang_2);
    cos_a2 = std::cos(ang_2);

    bx0 = cos_a2 * cos_a3 * B0;
    by0 = cos_a2 * sin_a3 * B0;
    bz0 = sin_a2 * B0;
  }
  
  // Set a uniform, steady gas.
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->u(IDN, k, j, i) = uniform_density;
        phydro->u(IM1, k, j, i) = ux0 * uniform_density;
        phydro->u(IM2, k, j, i) = uy0 * uniform_density;
        phydro->u(IM3, k, j, i) = uz0 * uniform_density;
        if (MAGNETIC_FIELDS_ENABLED) {
          if (NON_BAROTROPIC_EOS)
            phydro->u(IEN, k, j, i) = pres / (peos->GetGamma() - 1.0) +
                                      0.5 * B0 * B0 +
                                      0.5 *
                                          (SQR(phydro->u(IM1, k, j, i)) +
                                           SQR(phydro->u(IM2, k, j, i)) +
                                           SQR(phydro->u(IM3, k, j, i))) /
                                          phydro->u(IDN, k, j, i);
            pfield->b.x1f(k, j, i) = bx0;
            pfield->b.x2f(k, j, i) = by0;
            pfield->b.x3f(k, j, i) = bz0;
            if (i == ie) pfield->b.x1f(k, j, ie + 1) = bx0;
            if (j == je) pfield->b.x2f(k, je + 1, i) = by0;
            if (k == ke) pfield->b.x3f(ke + 1, j, i) = bz0;
        } else {
          if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN, k, j, i) =
                pres / (peos->GetGamma() - 1.0) +
                0.5 *
                    (SQR(phydro->u(IM1, k, j, i)) +
                     SQR(phydro->u(IM2, k, j, i)) +
                     SQR(phydro->u(IM3, k, j, i))) /
                    phydro->u(IDN, k, j, i);
          }
        }
      }
    }
  }

  // Get the position and velocity of the particle.
  if (PARTICLES != 0) {
    Real xp0, yp0, zp0, vpx0, vpy0, vpz0;
    xp0 = pin->GetOrAddReal("problem", "xp0", 0.0);
    yp0 = pin->GetOrAddReal("problem", "yp0", 0.0);
    zp0 = pin->GetOrAddReal("problem", "zp0", 0.0);
    vpx0 = pin->GetOrAddReal("problem", "vpx0", 0.0);
    vpy0 = pin->GetOrAddReal("problem", "vpy0", 0.0);
    vpz0 = pin->GetOrAddReal("problem", "vpz0", 0.0);

    // Check if the particle is in the meshblock.
    bool flag = true;
    if (block_size.nx1 > 1)
      flag = flag && block_size.x1min <= xp0 && xp0 < block_size.x1max;
    if (block_size.nx2 > 1)
      flag = flag && block_size.x2min <= yp0 && yp0 < block_size.x2max;
    if (block_size.nx3 > 1)
      flag = flag && block_size.x3min <= zp0 && zp0 < block_size.x3max;

    // Assign the particle, if any.
    if (flag) {
      ppar->npar = ppar->GetTotalTypeNumber();
      if (ppar->GetTotalTypeNumber() > ppar->nparmax)
        ppar->UpdateCapacity(ppar->GetTotalTypeNumber());
      for (std::size_t i = 0; i < ppar->GetTotalTypeNumber(); ++i) {
        ppar->xp(i) = xp0;
        ppar->yp(i) = yp0;
        ppar->zp(i) = zp0;
        ppar->vpx(i) = vpx0;
        ppar->vpy(i) = vpy0;
        ppar->vpz(i) = vpz0;
        if (PARTICLES == CHARGED_PAR) {
          ppar->tid(i) = i;
          //ppar->inv_f0(i) = 0.0;
        }
      }
    } else {
      ppar->npar = 0;
    }
  }
}
