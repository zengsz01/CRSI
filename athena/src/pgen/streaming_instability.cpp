//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file streaming_instability.cpp
//  \brief sets up a linear mode for the streaming instability between gas and particles.

// NOTE: In this setup, Y <-> Z.

// C++ standard libraries
#include <cmath>  // round()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "../parameter_input.hpp"
#include "../particles/particles.hpp"

#if PARTICLES != DUST_PAR
#error "This problem generator requires dust particles"
#endif

// Global parameters
static Real omega = 1.0;
static Real two_omega = 2.0 * omega;
static Real omega_half = 0.5 * omega;
static Real gas_accel_x = 0.0;

//======================================================================================
//! \fn void SourceTermsForGas(MeshBlock *pmb, const Real time, const Real dt,
//               const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
//               AthenaArray<Real> &cons) {
//  \brief Adds source terms to the gas.
//======================================================================================
void SourceTermsForGas(MeshBlock *pmb, const Real time, const Real dt,
                       const AthenaArray<Real> &prim,
                       const AthenaArray<Real> &prim_scalar,
                       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                       AthenaArray<Real> &cons_scalar) {
  // Apply the Coriolis and centrifugal forces, and linear gravity from the star, and
  // the background radial pressure gradient.
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real rho_dt = prim(IDN,k,j,i) * dt;
        cons(IM1,k,j,i) += rho_dt * (two_omega * prim(IVZ,k,j,i) + gas_accel_x);
        cons(IM3,k,j,i) -= rho_dt * omega_half * prim(IVX,k,j,i);
      }
    }
  }
}

//======================================================================================
//! \fn void DustParticles::UserSourceTerms(Real t, Real dt,
//                              const AthenaArray<Real>& meshsrc)
//  \brief Adds source terms to the particles.
//======================================================================================
void DustParticles::UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  // Apply the Coriolis and centrifugal forces, and linear gravity from the star.
  Real cx = dt * two_omega, cz = dt * omega_half;
  for (int k = 0; k < npar; ++k) {
    vpx(k) += cx * wz(k);
    vpz(k) -= cz * wx(k);
  }
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Initializes problem-specific data in Mesh class.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Preprocess constants.
  Real cs0 = pin->GetReal("hydro", "iso_sound_speed");
  Real duy0 = cs0 * pin->GetReal("problem", "duy0");
  omega = pin->GetOrAddReal("problem", "omega", omega);
  two_omega = 2.0 * omega;
  omega_half = 0.5 * omega;
  gas_accel_x = two_omega * duy0;

  // Enroll source terms.
  EnrollUserExplicitSourceFunction(SourceTermsForGas);
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Get the dust-to-gas density ratio.
  Real dtog = pin->GetReal("problem", "dtog");

  // Get the dimensionless stopping time.
  Real taus = omega * pin->GetReal("particles", "taus");

  // Get the wave number.
  Real cs0 = pin->GetReal("hydro", "iso_sound_speed");
  Real duy0 = cs0 * pin->GetReal("problem", "duy0");
  Real length = duy0 / omega;
  Real kx = pin->GetOrAddReal("problem", "kx", 0.0) / length,
       kz = pin->GetOrAddReal("problem", "kz", 0.0) / length;

  // Find the Nakagawa-Sekiya-Hayashi (1986) equilibrium solution.
  Real v = duy0 / (std::pow(1.0 + dtog, 2) + std::pow(taus, 2));
  Real ux0 = 2.0 * dtog * taus * v,
       uy0 = -((1.0 + dtog) + std::pow(taus, 2)) * v,
       vpx0 = -2.0 * taus * v,
       vpy0 = -(1.0 + dtog) * v;

  // Perturb the gas.
  Real amp = pin->GetOrAddReal("problem", "amplitude", 1e-6), dv = amp * duy0;
  Real drhog_re = amp * pin->GetOrAddReal("problem", "drhog_re", 0.0),
       drhog_im = amp * pin->GetOrAddReal("problem", "drhog_im", 0.0),
       dux_re = dv * pin->GetOrAddReal("problem", "dux_re", 0.0),
       dux_im = dv * pin->GetOrAddReal("problem", "dux_im", 0.0),
       duy_re = dv * pin->GetOrAddReal("problem", "duy_re", 0.0),
       duy_im = dv * pin->GetOrAddReal("problem", "duy_im", 0.0),
       duz_re = dv * pin->GetOrAddReal("problem", "duz_re", 0.0),
       duz_im = dv * pin->GetOrAddReal("problem", "duz_im", 0.0);

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      Real coskz = std::cos(kz * pcoord->x2v(j)), sinkz = std::sin(kz * pcoord->x2v(j));
      for (int i = is; i <= ie; ++i) {
        Real coskx = std::cos(kx * pcoord->x1v(i)), sinkx = std::sin(kx * pcoord->x1v(i));
        Real rhog = 1.0 + (drhog_re * coskx - drhog_im * sinkx) * coskz;
        phydro->u(IDN,k,j,i) = rhog;
        phydro->u(IM1,k,j,i) = rhog * (ux0 + (dux_re * coskx - dux_im * sinkx) * coskz);
        phydro->u(IM3,k,j,i) = rhog * (uy0 + (duy_re * coskx - duy_im * sinkx) * coskz);
        phydro->u(IM2,k,j,i) = -rhog * (duz_im * coskx + duz_re * sinkx) * sinkz;
      }
    }
  }

  int npar(0);
  if (PARTICLES == DUST_PAR) {
    // Find the total number of particles in each direction.
    RegionSize &mesh_size = pmy_mesh->mesh_size;
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
    npar = ppar->npar = npx1_loc * npx2_loc * npx3_loc;
    if (npar > ppar->nparmax) ppar->UpdateCapacity(npar);

    // Uniformly distribute the particles.
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
          ppar->vpy(ipar) = 0.0;
          ppar->vpz(ipar) = vpy0;
          ++ipar;
        }
      }
    }
  }
  if (kx == 0.0 && kz == 0.0) return;

  if (PARTICLES == DUST_PAR) {
    // Perturb the particle position.
    Real a = 0.5 * amp / (kx * kx + kz * kz);
    for (int k = 0; k < npar; ++k) {
      Real arg1 = kx * ppar->xp(k) + kz * ppar->yp(k),
           arg2 = kx * ppar->xp(k) - kz * ppar->yp(k);
      Real d = amp * std::sin(2.0 * arg1);
      ppar->xp(k) -= a * kx * (std::sin(arg1) + std::sin(arg2) - d);
      ppar->yp(k) -= a * kz * (std::sin(arg1) - std::sin(arg2) - d);
    }

    // Perturb the particle velocity.
    Real dvpx_re = dv * pin->GetOrAddReal("problem", "dvpx_re", 0.0),
         dvpx_im = dv * pin->GetOrAddReal("problem", "dvpx_im", 0.0),
         dvpy_re = dv * pin->GetOrAddReal("problem", "dvpy_re", 0.0),
         dvpy_im = dv * pin->GetOrAddReal("problem", "dvpy_im", 0.0),
         dvpz_re = dv * pin->GetOrAddReal("problem", "dvpz_re", 0.0),
         dvpz_im = dv * pin->GetOrAddReal("problem", "dvpz_im", 0.0);
    for (int k = 0; k < npar; ++k) {
      Real coskx = std::cos(kx * ppar->xp(k)),
           sinkx = std::sin(kx * ppar->xp(k));
      Real coskz = std::cos(kz * ppar->yp(k)),
           sinkz = std::sin(kz * ppar->yp(k));
      ppar->vpx(k) += (dvpx_re * coskx - dvpx_im * sinkx) * coskz;
      ppar->vpz(k) += (dvpy_re * coskx - dvpy_im * sinkx) * coskz;
      ppar->vpy(k) -= (dvpz_im * coskx + dvpz_re * sinkx) * sinkz;
    }
  }
}
