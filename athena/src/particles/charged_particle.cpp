//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//======================================================================================
//! \file charged_particles.cpp
//  \brief implements functions in the ChargedParticles class

// C++ headers
#include <algorithm>  // std::min(), std::min_element()
#include <random>     // mt19937, normal_distribution, uniform_real_distribution
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../defs.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../bvals/bvals.hpp"
#include "particles.hpp"


namespace {
Real c_reciprocal = 0;
Real c_reciprocal_square = 0;
std::vector<Real> q_over_m_over_c;
std::vector<Real> q_over_c;

// For phase randomization
std::mt19937_64 rng_generator;
#pragma omp threadprivate(rng_generator)
#pragma omp declare simd simdlen(SIMD_WIDTH)
Real GetLorentzFactor(const Real &v_x, const Real &v_y, const Real &v_z);
#pragma omp declare simd simdlen(SIMD_WIDTH)
void GetVelocity(const Real &v_x, const Real &v_y, const Real &v_z, Real &u_x,
                 Real &u_y, Real &u_z);
#pragma omp declare simd simdlen(SIMD_WIDTH)
Real GetEnergy(const Real &v_x, const Real &v_y, const Real &v_z);
}  // namespace

// Class variable initialization
bool ChargedParticles::initialized_ = false;
bool ChargedParticles::phase_random_ = false;
Real ChargedParticles::cfl_rot_ = 0.1;
Real ChargedParticles::c_ = HUGE_NUMBER;
std::vector<Real> ChargedParticles::mass_ = std::vector<Real>();
std::vector<Real> ChargedParticles::q_ = std::vector<Real>();

//--------------------------------------------------------------------------------------
//! \fn void ChargedParticles::Initialize(Mesh *pm, ParameterInput *pin)
//  \brief initializes the class.

void ChargedParticles::Initialize(Mesh *pm, ParameterInput *pin) {
  // Initialize firstly the parent class.
  Particles::Initialize(pm, pin);

  if (!initialized_) {
    Real c_tmp(HUGE_NUMBER), q_tmp(0.0), mass_tmp(1.0);
    Real cfl_par_tmp(pm->cfl_number), cfl_rotation_tmp(0.1);
    bool backreaction_tmp(false), delta_f_tmp(false), phase_random_tmp(false);
    InputBlock *pib = pin->pfirst_block;

    while (pib != nullptr) {
      if (pib->block_name.compare(0, 9, "particles") == 0) {
        // Read the CFL number for particles.
        cfl_par_tmp =
            pin->GetOrAddReal(pib->block_name, "cfl_par", cfl_par_tmp);
        // Read the CFL rotation number for particles.
        cfl_rotation_tmp =
            pin->GetOrAddReal(pib->block_name, "cfl_rot", cfl_rotation_tmp);
        // Read the speed of light.
        c_tmp = pin->GetOrAddReal(pib->block_name, "speed_of_light", c_tmp);

        // Define mass.
        mass_tmp = pin->GetOrAddReal(pib->block_name, "mass", mass_tmp);

        // Define electric charge.
        if (pin->DoesParameterExist(pib->block_name, "charge_over_mass_over_c")) {
          q_tmp = pin->GetReal(pib->block_name, "charge_over_mass_over_c") *
                  mass_tmp * c_tmp;
        } else {
          if (pin->DoesParameterExist(pib->block_name, "charge_over_mass"))
            q_tmp =
                pin->GetReal(pib->block_name, "charge_over_mass") * mass_tmp;
          else
            q_tmp = pin->GetOrAddReal(pib->block_name, "charge", q_tmp);
        }
        // Read backreaction.
        backreaction_tmp = pin->GetOrAddBoolean(pib->block_name, "backreaction",
                                                backreaction_tmp);
        delta_f_tmp = pin->GetOrAddBoolean(pib->block_name, "delta_f_enable",
                                           delta_f_tmp);
        phase_random_tmp = pin->GetOrAddBoolean(pib->block_name,
                                                "phase_random",
                                                phase_random_tmp);
        if (q_tmp == 0.0 || c_tmp == HUGE_NUMBER)
          backreaction_tmp = false;

        
        mass_.push_back(mass_tmp);
        q_.push_back(q_tmp);
        Particles::backreaction = backreaction_tmp || Particles::backreaction;
        delta_f_enable = delta_f_enable || delta_f_tmp;
        phase_random_ = phase_random_ || phase_random_tmp;
        cfl_par = std::min(cfl_par, cfl_par_tmp);
        cfl_rot_ = std::min(cfl_rot_, cfl_rotation_tmp);
      }
      pib = pib->pnext;  // move to next input block name
    }

    // If there are no particle blocks which have been defined
    if (mass_.size() == 0 && q_.size() == 0) {
      cfl_par = pin->GetOrAddReal("particles", "cfl_par", pm->cfl_number);
      cfl_rot_ = pin->GetOrAddReal("particles", "cfl_rot", 0.1);
      c_tmp = pin->GetOrAddReal("particles", "speed_of_light", c_tmp);
      mass_tmp = pin->GetOrAddReal("particles", "mass", mass_tmp);
      q_tmp = pin->GetOrAddReal("particles", "charge", q_tmp);
      backreaction_tmp =
          pin->GetOrAddBoolean("particles", "backreaction", false);
      delta_f_tmp =
          pin->GetOrAddBoolean("particles", "delta_f_enable", false);

      mass_.push_back(mass_tmp);
      q_.push_back(q_tmp);
      Particles::backreaction = backreaction_tmp;
      delta_f_enable = delta_f_tmp;
    }

    // Define the last input value as the speed of light.
    c_ = c_tmp;
    //if (!backreaction_) delta_f_enable = false;

    // Intitialize global variables
    c_reciprocal = 1.0 / c_;
    c_reciprocal_square = SQR(c_reciprocal);
    for (std::size_t i = 0; i < mass_.size(); ++i) {
      q_over_c.push_back(q_[i] * c_reciprocal);
      q_over_m_over_c.push_back(q_[i] / mass_[i] * c_reciprocal);
    }

     // Add particle inv_f0.
    if (delta_f_enable) {
      if0 = AddRealProperty();
      idelta_f_weight = AddWorkingArray();
    }

    // Initilize random seed for phase randomization
    if (phase_random_) {
      std::int64_t rseed(pin->GetOrAddInteger("problem", "random_seed", -1));
      if (rseed < 0) {
        std::random_device device;
        rseed = static_cast<std::int64_t>(device());
      }
      rng_generator.seed(rseed);
    }

    if (Particles::backreaction) {
      ParticleMesh::AddWeight();
      imvpx = ParticleMesh::AddMeshAux();
      imvpy = ParticleMesh::AddMeshAux();
      imvpz = ParticleMesh::AddMeshAux();
      imep = ParticleMesh::AddMeshAux();
    }

    initialized_ = true;
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn ChargedParticles::ChargedParticles(MeshBlock *pmb, ParameterInput *pin)
//  \brief constructs a ChargedParticles instance.

ChargedParticles::ChargedParticles(MeshBlock *pmb, ParameterInput *pin)
    : Particles(pmb, pin) {
  InputBlock *pib = pin->pfirst_block;
  int nparmax_tmp(0);

  while (pib != nullptr) {
    if (pib->block_name.compare(0, 9, "particles") == 0) {
      // Read nparmax.
      nparmax_tmp =
          pin->GetOrAddInteger(pib->block_name, "nparmax", nparmax_tmp);

      nparmax += nparmax_tmp;
    }
    pib = pib->pnext;  // move to next input block name
  }
  UpdateCapacity(nparmax);

  // Assign shorthands (need to do this for every constructor of a derived
  // class)
  AssignShorthands();

#pragma ivdep
  std::fill(&Particles::tid(0), &Particles::tid(nparmax), 0);
  if (Particles::delta_f_enable) {
#pragma ivdep
    std::fill(&inv_f0(0), &inv_f0(nparmax), 0.0);
  }

  field_.NewAthenaArray(InfoParticleMesh::nprop_bg, pmy_block->ncells3,
                        pmy_block->ncells2, pmy_block->ncells1);

  if (Particles::backreaction) {
    dpx1_.InitWithShallowSlice(ppm->meshaux, 4, imvpx, 1);
    dpx2_.InitWithShallowSlice(ppm->meshaux, 4, imvpy, 1);
    dpx3_.InitWithShallowSlice(ppm->meshaux, 4, imvpz, 1);
    dpe_.InitWithShallowSlice(ppm->meshaux, 4, imep, 1);
  }
}

//--------------------------------------------------------------------------------------
//! \fn ChargedParticles::~ChargedParticles()
//  \brief destroys a ChargedParticles instance.

ChargedParticles::~ChargedParticles() {
  field_.DeleteAthenaArray();

  if (Particles::backreaction) {
    dpx1_.DeleteAthenaArray();
    dpx2_.DeleteAthenaArray();
    dpx3_.DeleteAthenaArray();
    dpe_.DeleteAthenaArray();
  }
}

//--------------------------------------------------------------------------------------
//! \fn Real ChargedParticles::NewBlockTimeStep();
//  \brief returns the time step required by particles in the block.

Real ChargedParticles::NewBlockTimeStep() const {
  // Find the maximum coordinate speed.
  Real dt_inv2_maxv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real dt_inv2(0), vx1_mesh(0), vx2_mesh(0), vx3_mesh(0), vpx1(0), vpx2(0),
      vpx3(0), dt_inv2_max(0), gamma_square(0), dxf_rep(0), mag_square(0),
      q_over_m_c_square(0), period_rot_max(0);
  const Real cfl_par_inv(1.0 / cfl_par / cfl_par),
      cfl_rot_inv(1.0 / cfl_rot_ / cfl_rot_);
  int kkk(0), kk_max(0), pos_x1(0), pos_x2(0), pos_x3(0);

  // Initialize
#pragma ivdep
  std::fill(&dt_inv2_maxv[0], &dt_inv2_maxv[SIMD_WIDTH], 0.0);


  for (int k = 0; k < npar; k += SIMD_WIDTH) {
    kk_max = std::min(SIMD_WIDTH, npar - k);
#pragma omp simd simdlen(SIMD_WIDTH) private(                                \
    dt_inv2, vx1_mesh, vx2_mesh, vx3_mesh, vpx1, vpx2, vpx3, pos_x1, pos_x2, \
    pos_x3, gamma_square, mag_square, q_over_m_c_square, dxf_rep, kkk)
    for (int kk = 0; kk < kk_max; ++kk) {
      kkk = k + kk;
      q_over_m_c_square = SQR(q_over_m_over_c[Particles::tid(kkk)]);

      // Restrict dt by cfl_par
      vpx1 = vpx(kkk);
      vpx2 = vpy(kkk);
      vpx3 = vpz(kkk);
      pos_x1 = static_cast<int>(xi1(kkk));
      pos_x2 = static_cast<int>(xi2(kkk));
      pos_x3 = static_cast<int>(xi3(kkk));

      gamma_square =
          (SQR(vpx1) + SQR(vpx2) + SQR(vpx3)) * c_reciprocal_square + 1.0;
      /*pmy_block->pcoord->CartesianToMeshCoordsVector(xp(kkk), yp(kkk), zp(kkk),
                                                     vpx1, vpx2, vpx3, vx1_mesh,
                                                     vx2_mesh, vx3_mesh);

      if (active1) {
        dxf_rep =
            1.0 / pmy_block->pcoord->GetEdge1Length(pos_x3, pos_x2, pos_x1);
        dt_inv2 += SQR(vx1_mesh * dxf_rep);
      }
      if (active2) {
        dxf_rep =
            1.0 / pmy_block->pcoord->GetEdge2Length(pos_x3, pos_x2, pos_x1);
        dt_inv2 += SQR(vx2_mesh * dxf_rep);
      }
      if (active3) {
        dxf_rep =
            1.0 / pmy_block->pcoord->GetEdge3Length(pos_x3, pos_x2, pos_x1);
        dt_inv2 += SQR(vx3_mesh * dxf_rep);
      }*/
      
      pmy_block->pcoord->MeshCoordsToCartesian(
          pmy_block->pcoord->dx1f(pos_x1), pmy_block->pcoord->dx2f(pos_x2),
          pmy_block->pcoord->dx3f(pos_x3), vx1_mesh, vx2_mesh, vx3_mesh);

      if (active3) {
        dt_inv2 =
            SQR(vpx3 / vx3_mesh) + SQR(vpx2 / vx2_mesh) + SQR(vpx1 / vx1_mesh);
      } else if (active2) {
        dt_inv2 = SQR(vpx2 / vx2_mesh) + SQR(vpx1 / vx1_mesh);
      } else {
        dt_inv2 = SQR(vpx1 / vx1_mesh);
      }

      // Restrict dt by cfl_rot_ mag_square
      if (MAGNETIC_FIELDS_ENABLED)
        mag_square = SQR(pmy_block->pfield->bcc(IB1, pos_x3, pos_x2, pos_x1)) +
                     SQR(pmy_block->pfield->bcc(IB2, pos_x3, pos_x2, pos_x1)) +
                     SQR(pmy_block->pfield->bcc(IB3, pos_x3, pos_x2, pos_x1));

      dt_inv2_maxv[kk] =
              std::max(dt_inv2_maxv[kk],
                       std::max(dt_inv2 * cfl_par_inv,
                                q_over_m_c_square * mag_square * cfl_rot_inv) /
                           gamma_square);
    }
  }

  for (int i = 0; i < SIMD_WIDTH; ++i) {
    dt_inv2_max = std::max(dt_inv2_maxv[i], dt_inv2_max);
  }

  // Return the time step constrained by the coordinate speed.
  return dt_inv2_max > 0.0 ? 1.0 / std::sqrt(dt_inv2_max) : HUGE_NUMBER;
}

//--------------------------------------------------------------------------------------
//! \fn Real ChargedParticles::Statistics(std::function<Real(Real, Real)> const
//! &func) const
//  \brief count the statistical value, using the function func(momentum, pitch
//  angle) to calculate

Real ChargedParticles::Statistics(
    std::function<Real(Real, Real)> const &func) const {
  Real mom(0), weight(1.0), vpx1(0), vpx2(0), vpx3(0);
  Real stat(0);
  const bool delta_f_enable_copy(delta_f_enable);

  for (int k = 0; k < npar; ++k) {
    vpx1 = vpx(k);
    vpx2 = vpy(k);
    vpx3 = vpz(k);
    // TODO(sxc18): the intel compiler call wrong PhaseDist()
    if (delta_f_enable_copy)
      weight =
          1.0 - PhaseDist(xp(k), yp(k), zp(k), vpx1, vpx2, vpx3) * inv_f0(k);

    // assuming x-dir is the guiding field direction
    mom = std::sqrt(SQR(vpx1) + SQR(vpx2) + SQR(vpx3));
    // TODO(sxc18): this loop cannot be vectorized due to func()
    stat += func(mom, vpx1 / mom) * weight * mass_[Particles::tid(k)];
  }

  return stat;
}

//--------------------------------------------------------------------------------------
//! \fn std::vector<Real> ChargedParticles::GetTypes()
//  \brief return types of particles.

std::vector<Real> ChargedParticles::GetTypes() const {
  if (delta_f_enable)
    return mass_;
  else
    return q_over_m_over_c;
}

//--------------------------------------------------------------------------------------
//! \fn Real ChargedParticles::GetTypes(const int &k)
//  \brief return types of particles.

Real ChargedParticles::GetTypes(const int &k) const {
  if (delta_f_enable)
    return mass_[Particles::tid(k)];
  else
    return q_over_m_over_c[Particles::tid(k)];
}

//--------------------------------------------------------------------------------------
//! \fn void ChargedParticles::Integrate(int step)
//  \brief updates all particle positions and velocities from t to t + dt, with
//  boris pusher.

//  This is a temporary boris pusher without subcycling. Particles are only
//  pushed once at the second(final) step of mhd seconder-order integrating (vl2
//  or rk2). The feedback of the first step is the current feedback and that of
//  the second step is the momentum feedback

void ChargedParticles::Integrate(int stage) {
  const Real dt(0.5 * pmy_mesh->dt);

  if (!MAGNETIC_FIELDS_ENABLED) return;

  switch (stage) {
  case 1:
    if (Particles::backreaction) AssignCurrent(1, pmy_mesh->time, dt);
    break;

  case 2:
    Real u_x(0), u_y(0), u_z(0);
#pragma omp simd simdlen(SIMD_WIDTH) private(u_x, u_y, u_z)
    for (int k = 0; k < npar; ++k) {
    // Compute the velocity from momentum
    GetVelocity(vpx(k), vpy(k), vpz(k), u_x, u_y, u_z);

    // Update locations in the first half step
    xp(k) += dt * u_x;
    if (active2) yp(k) += dt * u_y;
    if (active3) zp(k) += dt * u_z;
    }

    // Record particles' indices before Boris pusher
    GetPositionIndices(npar, xp, yp, zp, xi1, xi2, xi3);

    // Coordinate source term (operator split)
    pmy_block->pcoord->AddCoordTermsPar(CoordTermsDivergence::preliminary,
                                        pmy_mesh->time, dt, field_, npar, vpx,
                                        vpy, vpz);

    // Convert mesh's prims into Cartesian coord
    const int nx1(pmy_block->ncells1), nx2(pmy_block->ncells2),
        nx3(pmy_block->ncells3);

    for (int k = 0; k < nx3; ++k) {
      for (int j = 0; j < nx2; ++j) {
#pragma omp simd simdlen(SIMD_WIDTH)
        for (int i = 0; i < nx1; ++i) {
        Real pos_mesh_x(0), pos_mesh_y(0), pos_mesh_z(0);

        pmy_block->pcoord->IndicesToMeshCoords(
            static_cast<Real>(i), static_cast<Real>(j),
            static_cast<Real>(k), pos_mesh_x, pos_mesh_y, pos_mesh_z);

        pmy_block->pcoord->MeshCoordsToCartesianVector(
            pos_mesh_x, pos_mesh_y, pos_mesh_z,
            pmy_block->phydro->w(IVX, k, j, i),
            pmy_block->phydro->w(IVY, k, j, i),
            pmy_block->phydro->w(IVZ, k, j, i),
            field_(InfoParticleMesh::IVx, k, j, i),
            field_(InfoParticleMesh::IVy, k, j, i),
            field_(InfoParticleMesh::IVz, k, j, i));

        pmy_block->pcoord->MeshCoordsToCartesianVector(
            pos_mesh_x, pos_mesh_y, pos_mesh_z,
            pmy_block->pfield->bcc(IB1, k, j, i),
            pmy_block->pfield->bcc(IB2, k, j, i),
            pmy_block->pfield->bcc(IB3, k, j, i),
            field_(InfoParticleMesh::IBx, k, j, i),
            field_(InfoParticleMesh::IBy, k, j, i),
            field_(InfoParticleMesh::IBz, k, j, i));
        }
      }
    }

    // Boris Pusher
    SourceTerms(pmy_mesh->time, 2.0 * dt, field_);  // field_ must in cartesian coord

    // Coordinate source term (operator split)
    pmy_block->pcoord->AddCoordTermsPar(CoordTermsDivergence::finale,
                                        pmy_mesh->time + dt, dt, field_, npar,
                                        vpx, vpy, vpz);

#pragma omp simd simdlen(SIMD_WIDTH) private(u_x, u_y, u_z)
    for (int k = 0; k < npar; ++k) {
    // Compute the velocity from momentum
    GetVelocity(vpx(k), vpy(k), vpz(k), u_x, u_y, u_z);

    // Update locations in the second half step
    xp(k) += dt * u_x;
    if (active2) yp(k) += dt * u_y;
    if (active3) zp(k) += dt * u_z;
    }

    // Update the position index.
    SetPositionIndices();
    break;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void ChargedParticles::UserSourceTerms(Real t, Real dt,
//                                          const AthenaArray<Real>& meshsrc)
//  \brief adds additional source terms to particles, overloaded by the user.

void __attribute__((weak))
ChargedParticles::UserSourceTerms(Real t, Real dt,
                                  const AthenaArray<Real> &meshsrc) {}

//--------------------------------------------------------------------------------------
//! \fn Real ChargedParticles::PhaseDist(const Real &x, const Real &y, const
//! Real &z, const Real &px, const Real &py, const Real &pz) const
//  \brief calculate distribution function in the phase space, overloaded by the
//  user.

Real __attribute__((weak))
ChargedParticles::PhaseDist(const Real &x, const Real &y, const Real &z,
                            const Real &px, const Real &py, const Real &pz) const {
  return 0.0;
}

//  \brief functions that will overloaded in the delta f method.

Real __attribute__((weak))
ChargedParticles::ChargeDen0(const Real &x, const Real &y, const Real &z) {
  return 0.0;
}

Real __attribute__((weak))
ChargedParticles::Jx0(const Real &x, const Real &y, const Real &z) {
  return 0.0;
}

Real __attribute__((weak))
ChargedParticles::Jy0(const Real &x, const Real &y, const Real &z) {
  return 0.0;
}

Real __attribute__((weak))
ChargedParticles::Jz0(const Real &x, const Real &y, const Real &z) {
  return 0.0;
}

Real __attribute__((weak))
ChargedParticles::DeltaMomx0(const Real &x, const Real &y, const Real &z) {
  return 0.0;
}

Real __attribute__((weak))
ChargedParticles::DeltaMomy0(const Real &x, const Real &y, const Real &z) {
  return 0.0;
}

Real __attribute__((weak))
ChargedParticles::DeltaMomz0(const Real &x, const Real &y, const Real &z) {
  return 0.0;
}

Real __attribute__((weak))
ChargedParticles::DeltaEnergy0(const Real &x, const Real &y, const Real &z) {
  return 0.0;
}

//--------------------------------------------------------------------------------------
//! \fn void ChargedParticles::XformFdbk2MeshCoord()
//  \brief transforming backreactions from Cartesian coord to mesh coord,
//  including for multi-level. This function is hard-coded.

void ChargedParticles::XformFdbk2MeshCoord() {
  if (!Particles::backreaction) return;
  Real pos_mesh_x(0), pos_mesh_y(0), pos_mesh_z(0);
  for (int k = 0; k < dpx1_.GetDim3(); ++k) {
    for (int j = 0; j < dpx1_.GetDim2(); ++j) {
#pragma omp simd simdlen(SIMD_WIDTH) private(pos_mesh_x, pos_mesh_y, pos_mesh_z)
      for (int i = 0; i < dpx1_.GetDim1(); ++i) {
        pmy_block->pcoord->IndicesToMeshCoords(
            static_cast<Real>(i), static_cast<Real>(j), static_cast<Real>(k),
            pos_mesh_x, pos_mesh_y, pos_mesh_z);
        pmy_block->pcoord->MeshCoordsToCartesian(pos_mesh_x, pos_mesh_y,
                                                 pos_mesh_z, pos_mesh_x,
                                                 pos_mesh_y, pos_mesh_z);
        pmy_block->pcoord->CartesianToMeshCoordsVector(
            pos_mesh_x, pos_mesh_y, pos_mesh_z, dpx1_(k, j, i), dpx2_(k, j, i),
            dpx3_(k, j, i), dpx1_(k, j, i), dpx2_(k, j, i), dpx3_(k, j, i));
      }
    }
  }
  if (pmy_mesh->multilevel) {
    const int mylevel = pmy_block->loc.level;
    for (int n = 0; n < pmy_block->pbval->nneighbor; ++n) {
      NeighborBlock &nb = pmy_block->pbval->neighbor[n];
      if (nb.snb.level == mylevel) {
        continue;
      } else if (nb.snb.level < mylevel) {  // To coarser
        AthenaArray<Real> &destination = ppm->pmbvar_.diff_level_var[nb.bufid];
        int si, sj, sk;
        const int cn(pmy_block->cnghost - 1);
        if (nb.ni.ox1 == 0) {
          std::int64_t &lx1 = pmy_block->loc.lx1;
          si = pmy_block->cis;
          if ((lx1 & 1LL) != 0LL) si -= cn;
        } else if (nb.ni.ox1 > 0) {
          si = pmy_block->cie + 1;
        } else
          si = pmy_block->cis - cn;
        if (nb.ni.ox2 == 0) {
          sj = pmy_block->cjs;
          if (pmy_block->block_size.nx2 > 1) {
            std::int64_t &lx2 = pmy_block->loc.lx2;
            if ((lx2 & 1LL) != 0LL) sj -= cn;
          }
        } else if (nb.ni.ox2 > 0) {
          sj = pmy_block->cje + 1;
        } else
          sj = pmy_block->cjs - cn;
        if (nb.ni.ox3 == 0) {
          sk = pmy_block->cks;
          if (pmy_block->block_size.nx3 > 1) {
            std::int64_t &lx3 = pmy_block->loc.lx3;
            if ((lx3 & 1LL) != 0LL) sk -= cn;
          }
        } else if (nb.ni.ox3 > 0) {
          sk = pmy_block->cke + 1;
        } else
          sk = pmy_block->cks - cn;
        // Copy from BoundaryValues::ProlongateBoundaries()
        for (int ck = 0; ck < destination.GetDim3(); ++ck) {
          const Real k((ck + sk - pmy_block->cks) * 2 + pmy_block->ks + 0.5);
          for (int cj = 0; cj < destination.GetDim2(); ++cj) {
            const Real j((cj + sj - pmy_block->cjs) * 2 + pmy_block->js + 0.5);
#pragma omp simd simdlen(SIMD_WIDTH) private(pos_mesh_x, pos_mesh_y, pos_mesh_z)
            for (int ci = 0; ci < destination.GetDim1(); ++ci) {
              const Real i((ci + si - pmy_block->cis) * 2 + pmy_block->is +
                           0.5);
              pmy_block->pcoord->IndicesToMeshCoords(i, j, k, pos_mesh_x,
                                                     pos_mesh_y, pos_mesh_z);
              pmy_block->pcoord->MeshCoordsToCartesian(pos_mesh_x, pos_mesh_y,
                                                       pos_mesh_z, pos_mesh_x,
                                                       pos_mesh_y, pos_mesh_z);
              pmy_block->pcoord->CartesianToMeshCoordsVector(
                  pos_mesh_x, pos_mesh_y, pos_mesh_z,
                  destination(imvpx, ck, cj, ci),
                  destination(imvpy, ck, cj, ci),
                  destination(imvpz, ck, cj, ci),
                  destination(imvpx, ck, cj, ci),
                  destination(imvpy, ck, cj, ci),
                  destination(imvpz, ck, cj, ci));
            }
          }
        }
      } else {  // To finer
        AthenaArray<Real> &destination = ppm->pmbvar_.diff_level_var[nb.bufid];
        int si, sj, sk;
        if (nb.ni.ox1 == 0) {
          si = pmy_block->is;
          if (nb.ni.fi1 == 1) si += pmy_block->block_size.nx1 / 2;
        } else if (nb.ni.ox1 > 0) {
          si = pmy_block->ie + 1;
        } else {
          si = pmy_block->is - NGHOST;
        }
        if (nb.ni.ox2 == 0) {
          sj = pmy_block->js;
          if (pmy_block->block_size.nx2 > 1) {
            if (nb.ni.ox1 != 0) {
              if (nb.ni.fi1 == 1) sj += pmy_block->block_size.nx2 / 2;
            } else {
              if (nb.ni.fi2 == 1) sj += pmy_block->block_size.nx2 / 2;
            }
          }
        } else if (nb.ni.ox2 > 0) {
          sj = pmy_block->je + 1;
        } else {
          sj = pmy_block->js - NGHOST;
        }
        if (nb.ni.ox3 == 0) {
          sk = pmy_block->ks;
          if (pmy_block->block_size.nx3 > 1) {
            if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
              if (nb.ni.fi1 == 1) sk += pmy_block->block_size.nx3 / 2;
            } else {
              if (nb.ni.fi2 == 1) sk += pmy_block->block_size.nx3 / 2;
            }
          }
        } else if (nb.ni.ox3 > 0) {
          sk = pmy_block->ke + 1;
        } else {
          sk = pmy_block->ks - NGHOST;
        }
        // Copy from CellCenteredBoundaryVariable::SetBoundaryFromFiner()
        for (int fk = 0; fk < destination.GetDim3(); ++fk) {
          const Real k(fk * 0.5 + sk - 0.25);
          for (int fj = 0; fj < destination.GetDim2(); ++fj) {
            const Real j(fj * 0.5 + sj - 0.25);
#pragma omp simd simdlen(SIMD_WIDTH) private(pos_mesh_x, pos_mesh_y, pos_mesh_z)
            for (int fi = 0; fi < destination.GetDim1(); ++fi) {
              const Real i(fi * 0.5 + si - 0.25);
              pmy_block->pcoord->IndicesToMeshCoords(i, j, k, pos_mesh_x,
                                                     pos_mesh_y, pos_mesh_z);
              pmy_block->pcoord->MeshCoordsToCartesian(pos_mesh_x, pos_mesh_y,
                                                       pos_mesh_z, pos_mesh_x,
                                                       pos_mesh_y, pos_mesh_z);
              pmy_block->pcoord->CartesianToMeshCoordsVector(
                  pos_mesh_x, pos_mesh_y, pos_mesh_z,
                  destination(imvpx, fk, fj, fi),
                  destination(imvpy, fk, fj, fi),
                  destination(imvpz, fk, fj, fi),
                  destination(imvpx, fk, fj, fi),
                  destination(imvpy, fk, fj, fi),
                  destination(imvpz, fk, fj, fi));
            }
          }
        }
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void ChargedParticles::ApplyBoundaryConditions()
//  \brief applies phase randomization

void ChargedParticles::ApplyLogicalConditions(const int &indice_for_new) {
  // Only apply logical conditions to those meshblocks locating at boundaries
  // Temporarily only supports periodic boundaries
  if (!(pmy_block->pbval->block_bcs[BoundaryFace::inner_x1] ==
            BoundaryFlag::periodic ||
        pmy_block->pbval->block_bcs[BoundaryFace::outer_x1] ==
            BoundaryFlag::periodic ||
        pmy_block->pbval->block_bcs[BoundaryFace::inner_x2] ==
            BoundaryFlag::periodic ||
        pmy_block->pbval->block_bcs[BoundaryFace::outer_x2] ==
            BoundaryFlag::periodic ||
        pmy_block->pbval->block_bcs[BoundaryFace::inner_x3] ==
            BoundaryFlag::periodic ||
        pmy_block->pbval->block_bcs[BoundaryFace::outer_x3] ==
            BoundaryFlag::periodic))
    return;

  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3, x10, x20, x30;
  Real vp1, vp2, vp3, vp10, vp20, vp30;
  bool flag(false);
  std::uniform_real_distribution<Real> udist(-PI * 0.5, PI * 0.5);
  for (int k = indice_for_new; k < npar; ++k) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    pcoord->CartesianToMeshCoords(xp0(k), yp0(k), zp0(k), x10, x20, x30);

    // For periodic conditions
    if (pmy_mesh->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic &&
        x1 < mesh_size.x1min) {
      x1 += mesh_size.x1max - mesh_size.x1min;
      x10 += mesh_size.x1max - mesh_size.x1min;
      flag = true;

    } else if (pmy_mesh->mesh_bcs[BoundaryFace::outer_x1] ==
                   BoundaryFlag::periodic &&
               x1 >= mesh_size.x1max) {
      x1 -= mesh_size.x1max - mesh_size.x1min;
      x10 -= mesh_size.x1max - mesh_size.x1min;
      flag = true;
    }
    if (active2) {
      if (pmy_mesh->mesh_bcs[BoundaryFace::inner_x2] ==
              BoundaryFlag::periodic &&
          x2 < mesh_size.x2min) {
        x2 += mesh_size.x2max - mesh_size.x2min;
        x20 += mesh_size.x2max - mesh_size.x2min;
        flag = true;
      } else if (pmy_mesh->mesh_bcs[BoundaryFace::outer_x2] ==
                     BoundaryFlag::periodic &&
                 x2 >= mesh_size.x2max) {
        x2 -= mesh_size.x2max - mesh_size.x2min;
        x20 -= mesh_size.x2max - mesh_size.x2min;
        flag = true;
      }
    }
    if (active3) {
      if (pmy_mesh->mesh_bcs[BoundaryFace::inner_x3] ==
              BoundaryFlag::periodic &&
          x3 < mesh_size.x3min) {
        x3 += mesh_size.x3max - mesh_size.x3min;
        x30 += mesh_size.x3max - mesh_size.x3min;
        flag = true;
      } else if (pmy_mesh->mesh_bcs[BoundaryFace::outer_x3] ==
                     BoundaryFlag::periodic &&
                 x3 >= mesh_size.x3max) {
        x3 -= mesh_size.x3max - mesh_size.x3min;
        x30 -= mesh_size.x3max - mesh_size.x3min;
        flag = true;
      }
    }

    if (flag) {
      pcoord->CartesianToMeshCoordsVector(xp(k), yp(k), zp(k), vpx(k), vpy(k),
                                          vpz(k), vp1, vp2, vp3);
      pcoord->CartesianToMeshCoordsVector(xp0(k), yp0(k), zp0(k), vpx0(k),
                                          vpy0(k), vpz0(k), vp10, vp20, vp30);
      pcoord->MeshCoordsToCartesian(x1, x2, x3, xp(k), yp(k), zp(k));
      pcoord->MeshCoordsToCartesian(x10, x20, x30, xp0(k), yp0(k), zp0(k));
      pcoord->MeshCoordsToCartesianVector(x1, x2, x3, vp1, vp2, vp3, vpx(k),
                                          vpy(k), vpz(k));
      pcoord->MeshCoordsToCartesianVector(x10, x20, x30, vp10, vp20, vp30,
                                          vpx0(k), vpy0(k), vpz0(k));
      // To randomize phase
      if (phase_random_ == true) {
        Real b_ang = 0.0;
        Real sin_ang = std::sin(b_ang);
        Real cos_ang = std::cos(b_ang);        
        Real phase_half(std::tan(udist(rng_generator)));
        Real v_0_x(vpx(k)), v_0_y(vpy(k)), v_0_z(vpz(k));
        Real temp(SQR(phase_half));
        // the dot product between b and v
        Real dot_temp = v_0_x * phase_half * cos_ang + v_0_z * phase_half * sin_ang;
        // Rotation due to B field
        Real v_1_x = - v_0_x * temp + v_0_y * sin_ang * phase_half + cos_ang * dot_temp * phase_half;
        Real v_1_y = - v_0_y * temp + v_0_z * phase_half * cos_ang - v_0_x * sin_ang * phase_half;
        Real v_1_z = - v_0_z * temp - v_0_y * phase_half * cos_ang + sin_ang * dot_temp * phase_half;
        temp = 2.0 / (1.0 + temp);  // \frac{2}{1+b^2}
        vpx(k) = v_0_x + temp * v_1_x;
        vpy(k) = v_0_y + temp * v_1_y;
        vpz(k) = v_0_z + temp * v_1_z;

        // Real phase_half(udist(rng_generator));
        // Real v_0_x(vpx(k)), v_0_y(vpy(k)), v_0_z(vpz(k));
        // Real temp(SQR(phase_half));
        // // the dot product between b and v
        // Real dot_temp = v_0_x * phase_half;
        // // Rotation due to B field
        // Real v_1_x = -v_0_x * temp + phase_half * dot_temp;
        // Real v_1_y = v_0_z * phase_half - v_0_y * temp;
        // Real v_1_z = -v_0_y * phase_half - v_0_z * temp;
        // temp = 2.0 / (1.0 + temp);  // \frac{2}{1+b^2}
        // // Push final half step due to E field
        // vpx(k) = v_0_x + temp * v_1_x;
        // vpy(k) = v_0_y + temp * v_1_y;
        // vpz(k) = v_0_z + temp * v_1_z;
      }
    }
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void ChargedParticles::DepositToMesh(const int &stage, Real t, Real dt,
//                          const AthenaArray<Real> &meshsrc, AthenaArray<Real>
//                          &meshdst)
//  \brief Deposits meshaux to Mesh.

void ChargedParticles::DepositToMesh(const int &stage, Real t, Real dt,
                                     const AthenaArray<Real> &meshsrc,
                                     AthenaArray<Real> &meshdst) {
  if (!Particles::backreaction) return;
  // Apply physical boundary after recieving boundary data but before calculate feedback
  pmy_block->pbval->ApplyPhysicalBoundaries(
      t, dt, std::vector<BoundaryVariable *>(1, &ppm->pmbvar_));
  

  const int ks(ppm->ks), ke(ppm->ke), js(ppm->js), je(ppm->je), is(ppm->is),
      ie(ppm->ie);
  Real pos_mesh_x(0), pos_mesh_y(0), pos_mesh_z(0);
  Real mesh_j_x(0), mesh_j_y(0), mesh_j_z(0), charge_den(0), b_x(0), b_y(0),
      b_z(0), e_x(0), e_y(0), e_z(0), u_x(0), u_y(0), u_z(0);

  // TODO(sxc18): Check following calculatation is whether valid in mesh coord
  switch (stage) { 
  case 1:
      if (MAGNETIC_FIELDS_ENABLED) {
        // Compute the current feedback in the mesh coords.
        for (int k = ks; k <= ke; ++k) {
          for (int j = js; j <= je; ++j) {
            if (delta_f_enable)
              pmy_block->pcoord->CellVolume(k, j, is, ie, ppm->vol);
#pragma omp simd simdlen(SIMD_WIDTH) private(                               \
    mesh_j_x, mesh_j_y, mesh_j_z, charge_den, b_x, b_y, b_z, e_x, e_y, e_z, \
    pos_mesh_x, pos_mesh_y, pos_mesh_z)
            for (int i = is; i <= ie; ++i) {
              pmy_block->pcoord->IndicesToMeshCoords(
                  static_cast<Real>(i), static_cast<Real>(j),
                  static_cast<Real>(k), pos_mesh_x, pos_mesh_y, pos_mesh_z);

              mesh_j_x = dpx1_(k, j, i);
              mesh_j_y = dpx2_(k, j, i);
              mesh_j_z = dpx3_(k, j, i);
              charge_den = ppm->weight(k, j, i);
              if (delta_f_enable) {
                mesh_j_x +=
                    Jx0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * ppm->vol(i);
                mesh_j_y +=
                    Jy0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * ppm->vol(i);
                mesh_j_z +=
                    Jz0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * ppm->vol(i);
                charge_den += ChargeDen0(pos_mesh_x, pos_mesh_y, pos_mesh_z) *
                              ppm->vol(i);
              }

              u_x = pmy_block->phydro->w(IVX, k, j, i);
              u_y = pmy_block->phydro->w(IVY, k, j, i);
              u_z = pmy_block->phydro->w(IVZ, k, j, i);
              b_x = pmy_block->pfield->bcc(IB1, k, j, i);
              b_y = pmy_block->pfield->bcc(IB2, k, j, i);
              b_z = pmy_block->pfield->bcc(IB3, k, j, i);

              // Compute E field from the frozen-in theorem
              e_x = u_z * b_y - u_y * b_z;
              e_y = u_x * b_z - u_z * b_x;
              e_z = u_y * b_x - u_x * b_y;

              // Current feedback in momentum equations
              dpx1_(k, j, i) =
                  -dt * (charge_den * e_x + mesh_j_y * b_z - mesh_j_z * b_y);
              dpx2_(k, j, i) =
                  -dt * (charge_den * e_y + mesh_j_z * b_x - mesh_j_x * b_z);
              dpx3_(k, j, i) =
                  -dt * (charge_den * e_z + mesh_j_x * b_y - mesh_j_y * b_x);

#if NON_BAROTROPIC_EOS
              // Current feedback in energy equations
              dpe_(k, j, i) =
                  -dt * (e_x * mesh_j_x + e_y * mesh_j_y + e_z * mesh_j_z);
#endif  // NON_BAROTROPIC_EOS
            }
          }
        }
      }
    break;

  case 2:
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        if (delta_f_enable)
          pmy_block->pcoord->CellVolume(k, j, is, ie, ppm->vol);
#pragma omp simd simdlen(SIMD_WIDTH) private( \
    b_x, b_y, b_z, e_x, e_y, e_z, pos_mesh_x, pos_mesh_y, pos_mesh_z)
        for (int i = is; i <= ie; ++i) {
          pmy_block->pcoord->IndicesToMeshCoords(
              static_cast<Real>(i), static_cast<Real>(j), static_cast<Real>(k),
              pos_mesh_x, pos_mesh_y, pos_mesh_z);
          if (delta_f_enable) {
//            dpx1_(k, j, i) -=
//                DeltaMomx0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * ppm->vol(i);
//            dpx2_(k, j, i) -=
//                DeltaMomy0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * ppm->vol(i);
//            dpx3_(k, j, i) -=
//                DeltaMomz0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * ppm->vol(i);
//
//#if NON_BAROTROPIC_EOS
//            dpe_(k, j, i) -=
//                DeltaEnergy0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * ppm->vol(i);
//#endif  // NON_BAROTROPIC_EOS

            if (MAGNETIC_FIELDS_ENABLED) {
              u_x = pmy_block->phydro->w(IVX, k, j, i);
              u_y = pmy_block->phydro->w(IVY, k, j, i);
              u_z = pmy_block->phydro->w(IVZ, k, j, i);
              b_x = pmy_block->pfield->bcc(IB1, k, j, i);
              b_y = pmy_block->pfield->bcc(IB2, k, j, i);
              b_z = pmy_block->pfield->bcc(IB3, k, j, i);

              // Compute E field from the frozen-in theorem
              e_x = u_z * b_y - u_y * b_z;
              e_y = u_x * b_z - u_z * b_x;
              e_z = u_y * b_x - u_x * b_y;

              // Current feedback in momentum equations
              const Real tmp(dt * ppm->vol(i));
              dpx1_(k, j, i) -=
                  tmp * (ChargeDen0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * e_x +
                         Jy0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * b_z -
                         Jz0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * b_y);
              dpx2_(k, j, i) -=
                  tmp * (ChargeDen0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * e_y +
                         Jz0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * b_x -
                         Jx0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * b_z);
              dpx3_(k, j, i) -=
                  tmp * (ChargeDen0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * e_z +
                         Jx0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * b_y -
                         Jy0(pos_mesh_x, pos_mesh_y, pos_mesh_z) * b_x);

#if NON_BAROTROPIC_EOS
              // Current feedback in energy equations
              dpe_(k, j, i) -=
                  tmp * (e_x * Jx0(pos_mesh_x, pos_mesh_y, pos_mesh_z) +
                         e_y * Jy0(pos_mesh_x, pos_mesh_y, pos_mesh_z) +
                         e_z * Jz0(pos_mesh_x, pos_mesh_y, pos_mesh_z));
#endif  // NON_BAROTROPIC_EOS
            }
          }
        }
      }
    }
    break;
  }

  // Deposit particle momentum(energy) changes to the gas.
  ppm->DepositMeshAux(meshdst, imvpx, IM1, InfoParticleMesh::nprop_aux);

  return;
}

void ChargedParticles::ReactToMeshAux(Real t, Real dt,
                                      const AthenaArray<Real> &meshsrc) {}

//--------------------------------------------------------------------------------------
//! \fn void ChargedParticles::AssignCurrent(const int &stage, Real t, Real dt)
//  \brief Assign the velocity onto each cell, and move forward a step

void ChargedParticles::AssignCurrent(const int &stage, Real t, Real dt) {
  int kkk(0), kk_max(0);
  ParticleMesh *pm_copy = ppm;
  Real weight(1.0);
  // Copy for vectorization (cheating the compiler)
  const bool delta_f_enable_copy(delta_f_enable),
      multilevel_copy(pmy_mesh->multilevel);

  // Initialize
  ppm->InitIntermAux(0.0);

  // Loop all the particles
  for (int k = 0; k < npar; k += SIMD_WIDTH) {
    kk_max = std::min(SIMD_WIDTH, npar - k);
#pragma omp simd simdlen(SIMD_WIDTH) private(kkk, weight)
    for (int kk = 0; kk < kk_max; ++kk) {
      kkk = k + kk;

      pm_copy->GetWeightTSC(Particles::xi1(kkk), Particles::xi2(kkk),
                            Particles::xi3(kkk), kk);

      // Record velocity of particle by deposit_var
      GetVelocity(vpx(kkk), vpy(kkk), vpz(kkk),
                  deposit_var[InfoParticleMesh::IPx][kk],
                  deposit_var[InfoParticleMesh::IPy][kk],
                  deposit_var[InfoParticleMesh::IPz][kk]);

      // Make preparation to  AssignParticlesToDifferentLevels
      if (pmy_mesh->multilevel) {
        vpx0(kkk) = deposit_var[InfoParticleMesh::IPx][kk];
        vpy0(kkk) = deposit_var[InfoParticleMesh::IPy][kk];
        vpz0(kkk) = deposit_var[InfoParticleMesh::IPz][kk];
      }

      // Record weight in the delta f method
      if (delta_f_enable_copy) {
        weight = 1.0 - PhaseDist(xp(kkk), yp(kkk), zp(kkk), vpx(kkk),
                                      vpy(kkk), vpz(kkk)) *
                                inv_f0(kkk);
        if (multilevel_copy) delta_f_weight(kkk) = weight;
      }
      pm_copy->DeltaFWeight(q_over_c[Particles::tid(kkk)] * weight, kk);
    }

    // Vectorize inside the callee ppm->Assign2IntermAux3D
    if (active3)
      for (int kk = 0; kk < kk_max; ++kk) ppm->Assign2IntermAuxAndWght3D(kk, 3);
     else if (active2)
      for (int kk = 0; kk < kk_max; ++kk) ppm->Assign2IntermAuxAndWght2D(kk, 3);
    else
      for (int kk = 0; kk < kk_max; ++kk) ppm->Assign2IntermAuxAndWght1D(kk, 3);
  }

  if (active3)
    ppm->AssignIntermAux3D2MeshAux(imvpx, InfoParticleMesh::IPx, 3,
                                   InfoParticleMesh::nprop_aux, true);
  else if (active2)
    ppm->AssignIntermAux2D2MeshAux(imvpx, InfoParticleMesh::IPx, 3,
                                   InfoParticleMesh::nprop_aux, true);
  else
    ppm->AssignIntermAux1D2MeshAux(imvpx, InfoParticleMesh::IPx, 3,
                                   InfoParticleMesh::nprop_aux, true);

  if (pmy_mesh->multilevel)
    ppm->AssignParticlesToDifferentLevels(
        auxprop, ivpx0, imvpx, 3, Particles::tid, q_over_c, delta_f_weight);
  // After AssignParticlesToDifferentLevels()
  XformFdbk2MeshCoord();
}

//--------------------------------------------------------------------------------------
//! \fn void ChargedParticles::SourceTerms(Real t, Real dt,
//                                   const AthenaArray<Real> &meshsrc)
//  \brief updates all particle velocities with Boris pusher. Interpolation and
//   assignment have been done at the same time. Pls see more from Bai et.
//   al(2015)

void ChargedParticles::SourceTerms(Real t, Real dt,
                                   const AthenaArray<Real> &meshsrc) {
  // Copy for vectorization (cheating the compiler)
  const bool backreaction_copy(Particles::backreaction),
      delta_f_enable_copy(delta_f_enable);
  Real *mass_vec_cpy;
  int err(0);  // check whether posix_memalign succeeds
  
  Real e_x(0), e_y(0), e_z(0), b_x(0), b_y(0), b_z(0), v_ini_x(0), v_ini_y(0),
      v_ini_z(0), v_0_x(0), v_0_y(0), v_0_z(0), dot_temp(0), temp(0), v_1_x(0),
      v_1_y(0), v_1_z(0);
  int kkk(0), kk_max(0);

  if (active3)
    ppm->InitIntermBkgnd3D(meshsrc);
  else if (active2)
    ppm->InitIntermBkgnd2D(meshsrc);
  else
    ppm->InitIntermBkgnd1D(meshsrc);

  if (backreaction_copy) {
    ppm->InitIntermAux(0.0);
  }

  // Loop over each particle

  // TODO(sxc18): implement for user source term

  for (int k = 0; k < npar; k += SIMD_WIDTH) {
    kk_max = std::min(SIMD_WIDTH, npar - k);

#pragma omp simd simdlen(SIMD_WIDTH) private(kkk)
    for (int kk = 0; kk < kk_max; ++kk) {
      kkk = k + kk;
      ppm->GetWeightTSC(Particles::xi1(kkk), Particles::xi2(kkk),
                        Particles::xi3(kkk), kk);
    }
    if (active3)
      for (int kk = 0; kk < kk_max; ++kk) ppm->Interpolate4IntermBkgnd3D(kk);
    else if (active2)
      for (int kk = 0; kk < kk_max; ++kk) ppm->Interpolate4IntermBkgnd2D(kk);
    else
      for (int kk = 0; kk < kk_max; ++kk) ppm->Interpolate4IntermBkgnd1D(kk);
#pragma omp simd simdlen(SIMD_WIDTH) private(                              \
    e_x, e_y, e_z, b_x, b_y, b_z, v_ini_x, v_ini_y, v_ini_z, v_0_x, v_0_y, \
    v_0_z, dot_temp, temp, v_1_x, v_1_y, v_1_z, kkk)
    for (int kk = 0; kk < kk_max; ++kk) {
      kkk = k + kk;

      const Real alpha(q_over_m_over_c[Particles::tid(kkk)] * dt * 0.5);

      // Compute E field from the frozen-in theorem.
      e_x =
          bkgnd[InfoParticleMesh::IVz][kk] * bkgnd[InfoParticleMesh::IBy][kk] -
          bkgnd[InfoParticleMesh::IVy][kk] * bkgnd[InfoParticleMesh::IBz][kk];
      e_y =
          bkgnd[InfoParticleMesh::IVx][kk] * bkgnd[InfoParticleMesh::IBz][kk] -
          bkgnd[InfoParticleMesh::IVz][kk] * bkgnd[InfoParticleMesh::IBx][kk];
      e_z =
          bkgnd[InfoParticleMesh::IVy][kk] * bkgnd[InfoParticleMesh::IBx][kk] -
          bkgnd[InfoParticleMesh::IVx][kk] * bkgnd[InfoParticleMesh::IBy][kk];

      e_x = alpha * e_x;
      e_y = alpha * e_y;
      e_z = alpha * e_z;
      v_ini_x = vpx(kkk);
      v_ini_y = vpy(kkk);
      v_ini_z = vpz(kkk);
      // Push half step due to E field
      v_0_x = v_ini_x + e_x;
      v_0_y = v_ini_y + e_y;
      v_0_z = v_ini_z + e_z;
      // Modified due to relativity effect
      temp = alpha / GetLorentzFactor(v_0_x, v_0_y, v_0_z);
      b_x = bkgnd[InfoParticleMesh::IBx][kk] * temp;
      b_y = bkgnd[InfoParticleMesh::IBy][kk] * temp;
      b_z = bkgnd[InfoParticleMesh::IBz][kk] * temp;  // the b in the Boris pusher
      temp = SQR(b_x) + SQR(b_y) + SQR(b_z);          // the b square
      // the dot product between b and v
      dot_temp = v_0_x * b_x + v_0_y * b_y + v_0_z * b_z;
      // Rotation due to B field
      v_1_x = v_0_y * b_z - v_0_z * b_y - v_0_x * temp + b_x * dot_temp;
      v_1_y = v_0_z * b_x - v_0_x * b_z - v_0_y * temp + b_y * dot_temp;
      v_1_z = v_0_x * b_y - v_0_y * b_x - v_0_z * temp + b_z * dot_temp;
      temp = 2.0 / (1.0 + temp);  // \frac{2}{1+b^2}
      // Push final half step due to E field
      v_1_x = v_0_x + temp * v_1_x + e_x;
      v_1_y = v_0_y + temp * v_1_y + e_y;
      v_1_z = v_0_z + temp * v_1_z + e_z;

      // Record final momentum
      vpx(kkk) = v_1_x;
      vpy(kkk) = v_1_y;
      vpz(kkk) = v_1_z;

      if (backreaction_copy) {
        dot_temp = mass_[Particles::tid(kkk)];
        // TODO(sxc18): CAUTION! The intel compiler(2020) could not go through this line.
        if (delta_f_enable_copy)
          dot_temp *=
              1.0 - PhaseDist(xp(kkk), yp(kkk), zp(kkk),
                              0.5 * (v_ini_x + v_1_x), 0.5 * (v_ini_y + v_1_y),
                              0.5 * (v_ini_z + v_1_z)) *
                        inv_f0(kkk);

        // Momentum losing of gas
        deposit_var[InfoParticleMesh::IPx][kk] = (v_ini_x - v_1_x);
        deposit_var[InfoParticleMesh::IPy][kk] = (v_ini_y - v_1_y);
        deposit_var[InfoParticleMesh::IPz][kk] = (v_ini_z - v_1_z);
#if NON_BAROTROPIC_EOS
        // Energy losing of gas
        deposit_var[InfoParticleMesh::IE][kk] =
            (GetEnergy(v_ini_x, v_ini_y, v_ini_z) -
             GetEnergy(v_1_x, v_1_y, v_1_z));
#endif  // NON_BAROTROPIC_EOS
        ppm->DeltaFWeight(dot_temp, kk);
        // Make prepare to  AssignParticlesToDifferentLevels
        if (pmy_mesh->multilevel) {
          vpx0(kkk) = deposit_var[InfoParticleMesh::IPx][kk] * dot_temp;
          vpy0(kkk) = deposit_var[InfoParticleMesh::IPy][kk] * dot_temp;
          vpz0(kkk) = deposit_var[InfoParticleMesh::IPz][kk] * dot_temp;
#if NON_BAROTROPIC_EOS
          ep0(kkk) = deposit_var[InfoParticleMesh::IE][kk] * dot_temp;
#endif  // NON_BAROTROPIC_EOS
        }
      }
    }

    if (backreaction_copy) {
      // Vectorize inside the callee ppm->Assign2IntermAux
      if (active3)
        for (int kk = 0; kk < kk_max; ++kk) ppm->Assign2IntermAux3D(kk);
      else if (active2)
        for (int kk = 0; kk < kk_max; ++kk) ppm->Assign2IntermAux2D(kk);
      else
        for (int kk = 0; kk < kk_max; ++kk) ppm->Assign2IntermAux1D(kk);
    }
  }

  if (backreaction_copy) {
    if (active3)
      ppm->AssignIntermAux3D2MeshAux(imvpx, InfoParticleMesh::IPx,
                                     InfoParticleMesh::nprop_aux,
                                     InfoParticleMesh::nprop_aux, false);
    else if (active2)
      ppm->AssignIntermAux2D2MeshAux(imvpx, InfoParticleMesh::IPx,
                                     InfoParticleMesh::nprop_aux,
                                     InfoParticleMesh::nprop_aux, false);
    else
      ppm->AssignIntermAux1D2MeshAux(imvpx, InfoParticleMesh::IPx,
                                     InfoParticleMesh::nprop_aux,
                                     InfoParticleMesh::nprop_aux, false);
    
    if (pmy_mesh->multilevel)
      ppm->AssignParticlesToDifferentLevels(auxprop, ivpx0, imvpx,
                                            InfoParticleMesh::nprop_aux);
    // After AssignParticlesToDifferentLevels()
    XformFdbk2MeshCoord();
  }
}

//--------------------------------------------------------------------------------------
//! \fn void ChargedParticles::SetParticleMesh(int step)
//  \brief receives ParticleMesh meshaux near boundaries from neighbors and
//  returns a
//         flag indicating if all receives are completed.

void ChargedParticles::SetParticleMesh(int stage) {
  if (!Particles::backreaction) return;
  // Flush ParticleMesh receive buffers.
  ppm->pmbvar_.SetBoundaries();
 
  Real dt(pmy_mesh->dt);

//  if (Particles::backreaction && MAGNETIC_FIELDS_ENABLED &&
//      !(stage == 2 && !delta_f_enable)) {
//    const int ks(ppm->ks), ke(ppm->ke), js(ppm->js), je(ppm->je), is(ppm->is),
//        ie(ppm->ie);
//    // field_ in mesh coord
//    for (int k = ks; k < ke; ++k)
//      for (int j = js; j < je; ++j)
//#pragma omp simd simdlen(SIMD_WIDTH)
//        for (int i = is; i < ie; ++i)
//          field_(InfoParticleMesh::IVx, k, j, i) =
//              pmy_block->phydro->w(IVX, k, j, i);
//
//    for (int k = ks; k < ke; ++k)
//      for (int j = js; j < je; ++j)
//#pragma omp simd simdlen(SIMD_WIDTH)
//        for (int i = is; i < ie; ++i)
//          field_(InfoParticleMesh::IVy, k, j, i) =
//              pmy_block->phydro->w(IVY, k, j, i);
//
//    for (int k = ks; k < ke; ++k)
//      for (int j = js; j < je; ++j)
//#pragma omp simd simdlen(SIMD_WIDTH)
//        for (int i = is; i < ie; ++i)
//          field_(InfoParticleMesh::IVz, k, j, i) =
//              pmy_block->phydro->w(IVZ, k, j, i);
//
//    for (int k = ks; k < ke; ++k)
//      for (int j = js; j < je; ++j)
//#pragma omp simd simdlen(SIMD_WIDTH)
//        for (int i = is; i < ie; ++i)
//          field_(InfoParticleMesh::IBx, k, j, i) =
//              pmy_block->pfield->bcc(IB1, k, j, i);
//
//    for (int k = ks; k < ke; ++k)
//      for (int j = js; j < je; ++j)
//#pragma omp simd simdlen(SIMD_WIDTH)
//        for (int i = is; i < ie; ++i)
//          field_(InfoParticleMesh::IBy, k, j, i) =
//              pmy_block->pfield->bcc(IB2, k, j, i);
//
//    for (int k = ks; k < ke; ++k)
//      for (int j = js; j < je; ++j)
//#pragma omp simd simdlen(SIMD_WIDTH)
//        for (int i = is; i < ie; ++i)
//          field_(InfoParticleMesh::IBz, k, j, i) =
//              pmy_block->pfield->bcc(IB3, k, j, i);
//  }
//  // TODO(sxc18): check whether we need to combine prim to field_
  switch (stage) {
    case 1:
      dt = 0.5 * pmy_mesh->dt;
      break;

    case 2:
      dt = pmy_mesh->dt;
      break;
  }
  DepositToMesh(stage, pmy_mesh->time, dt, field_, pmy_block->phydro->u);
  
}

namespace {
//--------------------------------------------------------------------------------------
//! \fn void GetVelocity(const Real &v_x, const Real &v_y, const Real &v_z,
//                        Real &u_x, Real &u_y, Real &u_z)
//  \brief returns the velocity {u_x, u_y, u_z} with a given momentum {v_x, v_y,
//  v_z}

inline void GetVelocity(const Real &v_x, const Real &v_y, const Real &v_z,
                        Real &u_x, Real &u_y, Real &u_z) {
  Real gamma_reciprocal(1.0 / GetLorentzFactor(v_x, v_y, v_z));
  u_x = v_x * gamma_reciprocal;
  u_y = v_y * gamma_reciprocal;
  u_z = v_z * gamma_reciprocal;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn Real GetLorentzFactor(const Real &v_x, const Real &v_y, const Real &v_z)
//  \brief returns the lorentz factor with a given momentum {v_x, v_y, v_z}

inline Real GetLorentzFactor(const Real &v_x, const Real &v_y,
                             const Real &v_z) {
  return std::sqrt((SQR(v_x) + SQR(v_y) + SQR(v_z)) * c_reciprocal_square +
                   1.0);
}

//--------------------------------------------------------------------------------------
//! \fn Real Real GetEnergy(const Real &v_x, const Real &v_y, const Real &v_z)
//  \brief returns the equivalent energy factor with a given momentum {v_x, v_y,
//  v_z}, pls see more from Bai. et al (2015)

inline Real GetEnergy(const Real &v_x, const Real &v_y, const Real &v_z) {
  Real temp(SQR(v_x) + SQR(v_y) + SQR(v_z));
  return temp / (std::sqrt(1.0 + temp * c_reciprocal_square) + 1.0);
}
}  // namespace