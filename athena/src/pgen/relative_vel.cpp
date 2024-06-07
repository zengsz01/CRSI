//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file relative_vel.cpp
//  \brief relative motion for electrons and positrons, in non-relativisitic limit.

/// C headers

// C++ headers
#include <algorithm>
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../particles/particles.hpp"

void IonNeutralDamp(MeshBlock *pmb, const Real time, const Real dt,
                    const AthenaArray<Real> &prim,
                    const AthenaArray<Real> &prim_scalar,
                    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                    AthenaArray<Real> &cons_scalar);

namespace {
constexpr Real uniform_density = 1.0;
Real dx1_reciprocal, dx2_reciprocal,
    dx3_reciprocal;               // Number density of charged particles
Real ux0(0), uy0(0), uz0(0);      // velocity of gas
std::vector<Real> vpx, vpy, vpz;  // velocity of different particles
Real nu_in;
std::size_t types(0);             // Number of species in the block "problem"
}  // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also
//  be used to initialize variables which are global to (and therefore can be
//  passed to) other functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  ux0 = pin->GetOrAddReal("problem", "ux0", 0.0);
  ;
  uy0 = pin->GetOrAddReal("problem", "uy0", 0.0);
  uz0 = pin->GetOrAddReal("problem", "uz0", 0.0);
  nu_in = pin->GetOrAddReal("problem", "nu_in", 0.0);
  if (nu_in > 0.0) {
    EnrollUserExplicitSourceFunction(IonNeutralDamp);
  }

  // Initialize charged particles' properties
  if (PARTICLES == CHARGED_PAR) {
    // Total number of particles = npx1 * npx2 * npx3
    int npx1((mesh_size.nx1 > 1)
                 ? pin->GetOrAddInteger("problem", "npx1", mesh_size.nx1)
                 : 1),
        npx2((mesh_size.nx2 > 1)
                 ? pin->GetOrAddInteger("problem", "npx2", mesh_size.nx2)
                 : 1),
        npx3((mesh_size.nx3 > 1)
                 ? pin->GetOrAddInteger("problem", "npx3", mesh_size.nx3)
                 : 1);
    dx1_reciprocal =
        static_cast<Real>(npx1) / (mesh_size.x1max - mesh_size.x1min);
    dx2_reciprocal =
        static_cast<Real>(npx2) / (mesh_size.x2max - mesh_size.x2min);
    dx3_reciprocal =
        static_cast<Real>(npx3) / (mesh_size.x3max - mesh_size.x3min);

    int i(0);
    while (pin->DoesParameterExist("problem", "vpx" + std::to_string(i)) ||
           pin->DoesParameterExist("problem", "vpy" + std::to_string(i)) ||
           pin->DoesParameterExist("problem", "vpz" + std::to_string(i))) {
      vpx.push_back(
          pin->GetOrAddReal("problem", "vpx" + std::to_string(i), 0.0));
      vpy.push_back(
          pin->GetOrAddReal("problem", "vpy" + std::to_string(i), 0.0));
      vpz.push_back(
          pin->GetOrAddReal("problem", "vpz" + std::to_string(i), 0.0));
      ++i;
    }
    types = i;

    // The total (gas+CR) momentum is zero
    for (std::size_t i = 0; i < types; ++i) {
      ux0 -= vpx[i];
      uy0 -= vpy[i];
      uz0 -= vpz[i];
    }
    Real mass_tmp(pin->GetOrAddReal("particles0", "mass", 1.0));
    ux0 *= dx1_reciprocal * dx2_reciprocal * dx3_reciprocal * mass_tmp /
           uniform_density;
    uy0 *= dx1_reciprocal * dx2_reciprocal * dx3_reciprocal * mass_tmp /
           uniform_density;
    uz0 *= dx1_reciprocal * dx2_reciprocal * dx3_reciprocal * mass_tmp /
           uniform_density;
  }
  return;
}

//========================================================================================
//! \fn ProblemGenerator
//  \brief circularly polarized Alfven wave problem generator for 1D/2D/3D
//  problems.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real Bx0(0), By0(0), Bz0(0);  // Uniform B field
  Real B0, pres;                // Uniform B field strength and thermal pressure
  if (NON_BAROTROPIC_EOS) {
    pres = pin->GetReal("problem", "pres");
  } else {
    Real iso_cs(peos->GetIsoSoundSpeed());
    pres = uniform_density * SQR(iso_cs);
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    Real beta(pin->GetReal("problem", "beta"));
    B0 = std::sqrt(static_cast<Real>(2.0 * pres / beta));
    /*int Bdir(pin->GetOrAddInteger("problem", "Bdir", 3));
    switch (static_cast<int>(std::abs(Bdir))) {
      case 1:
        Bx0 = SIGN(Bdir) * B0;
        break;
      case 2:
        By0 = SIGN(Bdir) * B0;
        break;
      case 3:
        Bz0 = SIGN(Bdir) * B0;
        break;
    }*/
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

    Bx0 = cos_a2 * cos_a3 * B0;
    By0 = cos_a2 * sin_a3 * B0;
    Bz0 = sin_a2 * B0;
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
          pfield->b.x1f(k, j, i) = Bx0;
          pfield->b.x2f(k, j, i) = By0;
          pfield->b.x3f(k, j, i) = Bz0;
          if (i == ie) pfield->b.x1f(k, j, ie + 1) = Bx0;
          if (j == je) pfield->b.x2f(k, je + 1, i) = By0;
          if (k == ke) pfield->b.x3f(ke + 1, j, i) = Bz0;
        } else {
          if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN, k, j, i) = pres / (peos->GetGamma() - 1.0) +
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

  if (PARTICLES == CHARGED_PAR) {
    // Determine number of particles in the block.
    int npx1_loc(static_cast<int>(
        std::round((block_size.x1max - block_size.x1min) * dx1_reciprocal))),
        npx2_loc(static_cast<int>(std::round(
            (block_size.x2max - block_size.x2min) * dx2_reciprocal))),
        npx3_loc(static_cast<int>(std::round(
            (block_size.x3max - block_size.x3min) * dx3_reciprocal)));
    int npar = ppar->npar =
        npx1_loc * npx2_loc * npx3_loc * ppar->GetTotalTypeNumber();
    if (npar > ppar->nparmax) ppar->UpdateCapacity(npar);

    // Assign the particles.

    if (npar > 0) {
      Real dx1(1.0 / dx1_reciprocal), dx2(1.0 / dx2_reciprocal),
          dx3(1.0 / dx3_reciprocal);
      int ipar = 0;
      for (int k = 0; k < npx3_loc; ++k) {
        Real zp1 = block_size.x3min + (k + 0.5) * dx3;
        for (int j = 0; j < npx2_loc; ++j) {
          Real yp1 = block_size.x2min + (j + 0.5) * dx2;
          for (int i = 0; i < npx1_loc; ++i) {
            Real xp1 = block_size.x1min + (i + 0.5) * dx1;
            for (std::size_t tid = 0;
                 tid < std::min(ppar->GetTotalTypeNumber(), types); ++tid) {
              ppar->xp(ipar) = xp1;
              ppar->yp(ipar) = yp1;
              ppar->zp(ipar) = zp1;
              ppar->vpx(ipar) = vpx[tid];
              ppar->vpy(ipar) = vpy[tid];
              ppar->vpz(ipar) = vpz[tid];
              ppar->tid(ipar) = tid;
              ++ipar;
            }
          }
        }
      }
    }
  }
  return;
}

void IonNeutralDamp(MeshBlock *pmb, const Real time, const Real dt,
                    const AthenaArray<Real> &prim,
                    const AthenaArray<Real> &prim_scalar,
                    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                    AthenaArray<Real> &cons_scalar) {
  const Real fac(nu_in * dt);
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real tmp(fac * prim(IDN, k, j, i));
        cons(IM1, k, j, i) -= tmp * prim(IVX, k, j, i);
        cons(IM2, k, j, i) -= tmp * prim(IVY, k, j, i);
        cons(IM3, k, j, i) -= tmp * prim(IVZ, k, j, i);
      }
    }
  }
}