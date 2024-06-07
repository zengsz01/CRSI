//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file reflect_shock.cpp
//  \brief Problem generator for particle acceleration in collisionless shocks.
//
// REFERENCE: X.-N.Bai+, "Magnetohydrodynamic-particle-in-cell Method for
// Coupling Cosmic Rays with a Thermal Plasma: Application to Non-relativistic
// Shocks", ApJ, 809, 55B (2015)

// C headers

// C++ headers
#include <chrono>
#include <cmath>      // sqrt(), sin(), cos(), min(), round()
#include <vector>
#include <algorithm>  // find(), erase()
#include <utility>    // pair
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <numeric>    // std::accumulate

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../defs.hpp"
#include "../mesh/mesh.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../utils/utils.hpp"  // ran2()
#include "../parameter_input.hpp"
#include "../particles/particles.hpp"
#include "../particles/particle-mesh.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace {
// Parameters which define initial solution
constexpr Real den = 1.0;
Real density_floor, pressure_floor;
Real sin_a, cos_a, b_guide, v_upstrm, pres, v_pert;
std::int64_t iseed_org;
// Parameters about particles' injection
Real cr_frac, energy_cr_inj, energy_thermal, t_remove;
std::vector<std::vector<std::pair<int, int>>> par_add; // particles added before t_remove
std::vector<std::pair<int, int>> par_remove;
// analysis properties
bool analysis_enable(false);
int nxbin, npbin;
Real analysis_t_inv, dpbin_inv, ln_p_min, energy_max;
Real *spec;
std::vector<Real> p_bin;
std::string spec_basename;
int RefinementCondition(MeshBlock *pmb);
Real HistroryCurrentXUpStream(MeshBlock *pmb, int iout);
Real HistroryNumUpStream(MeshBlock *pmb, int iout);
Real HistroryNBPerMesh(MeshBlock *pmb, int iout);
Real HistroryNParPerMesh(MeshBlock *pmb, int iout);
std::vector<Real> current_x;
std::vector<Real> num_up;
}  // namespace


void Mesh::InitUserMeshData(ParameterInput *pin) {
  iseed_org = std::abs(static_cast<std::int64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count()));
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &iseed_org, 1, MPI_INT64_T, MPI_PROD,
                MPI_COMM_WORLD);
#endif
  iseed_org = pin->GetOrAddInteger("problem", "random_seed", iseed_org);
  std::int64_t iseed(-1 - iseed_org);  // Initialize on the first call to ran2
  ran2(&iseed);
  b_guide = pin->GetReal("problem", "b_guide");
  Real ang(pin->GetOrAddReal("problem", "ang", 0.0));  // defalut is parallel shock
  sin_a = std::sin(ang);
  cos_a = std::cos(ang);
  v_upstrm = pin->GetReal("problem", "v_upstrm");
  if (v_upstrm <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
        << "[reflect_shock]: v_upstrm must be positive!" << std::endl;
    ATHENA_ERROR(msg);
  }
  v_pert = pin->GetReal("problem", "v_pert");
  if ((v_pert < 0.0) || (v_pert > 0.5 * b_guide / std::sqrt(den))) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
        << "[reflect_shock]: v_pert must be within [0 0.5*v_A]!" << std::endl;
    ATHENA_ERROR(msg);
  }
  pres = pin->GetReal("problem", "pres");

  if (adaptive) EnrollUserRefinementCondition(RefinementCondition);

  if (PARTICLES == CHARGED_PAR) {
    density_floor = pin->GetOrAddReal(
        "hydro", "dfloor", std::sqrt(1024 * std::numeric_limits<float>::min()));
    pressure_floor = pin->GetOrAddReal(
        "hydro", "pfloor", std::sqrt(1024 * std::numeric_limits<float>::min()));
    cr_frac = pin->GetReal("problem", "CR_frac");
    if ((cr_frac < 0.0) || (cr_frac > 0.1)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "[reflect_shock]: CR mass fraction should be within [0 0.1]!" << std::endl;
      ATHENA_ERROR(msg);
    }
    energy_cr_inj = pin->GetReal("problem", "ECR_inj");
    if (energy_cr_inj < 1.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "[reflect_shock]: CR energy should satisfy 1<=ECR_inj!"
          << std::endl;
      ATHENA_ERROR(msg);
    }
    if (energy_cr_inj * cr_frac  > 0.5) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "[reflect_shock]: total CR energy density = "
          << energy_cr_inj * cr_frac << " is too large!" << std::endl;
      ATHENA_ERROR(msg);
    }
    // Default is to not remove particles after t_remove
    t_remove = pin->GetOrAddReal("problem", "T_remove", -1.0);
    if (t_remove > tlim) t_remove = -1.0;

    energy_thermal = pin->GetOrAddReal("problem", "E_th", 1.0);
    energy_thermal *= 0.5 * SQR(v_upstrm);

    // Initialize analysis properties
    analysis_enable = pin->GetOrAddBoolean("analysis", "enable", false);
    if (analysis_enable) {
      spec_basename = pin->GetString("job", "problem_id");
      // Read parameters
      energy_max = pin->GetOrAddReal("analysis", "E_max", 2.0 * energy_cr_inj);
      energy_max *= 0.5 * SQR(v_upstrm);
      if (energy_thermal >= energy_max) {
        std::stringstream msg;
        msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]"
            << std::endl
            << "[reflect_shock]: E_max must be larger than E_th!" << std::endl;
        ATHENA_ERROR(msg);
      }
      npbin = pin->GetOrAddInteger("analysis", "npbin", 4);  // must > 2
      if (npbin <= 2) {
        std::stringstream msg;
        msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]"
            << std::endl
            << "[reflect_shock]: npbin must be greater than 4!" << std::endl;
        ATHENA_ERROR(msg);
      }
      nxbin = pin->GetOrAddInteger("analysis", "nxbin", 2);
      analysis_t_inv = 1.0 / pin->GetOrAddReal("analysis", "dt", t_remove);
      if (1.0 / analysis_t_inv < 0) analysis_enable = false;

      // Initialize
      spec = new Real[npbin * nxbin]();
      // In non-relativistic limit
      dpbin_inv = 2.0 * static_cast<Real>(npbin) /
                  std::log(energy_max / energy_thermal);
      ln_p_min = std::log(std::sqrt(2.0 * energy_thermal));
      for (int i = 0; i < npbin; ++i)
        p_bin.push_back(std::exp((i + 0.5) / dpbin_inv + ln_p_min));

      AllocateUserHistoryOutput(6);
      EnrollUserHistoryOutput(0, HistroryCurrentXUpStream, "Jx");
      EnrollUserHistoryOutput(1, HistroryNumUpStream, "N_up");
      EnrollUserHistoryOutput(2, HistroryNBPerMesh, "nblock_max",
                              UserHistoryOperation::max);
      EnrollUserHistoryOutput(3, HistroryNBPerMesh, "nblock_min",
                              UserHistoryOperation::min);
      EnrollUserHistoryOutput(4, HistroryNParPerMesh, "npar_max",
                              UserHistoryOperation::max);
      EnrollUserHistoryOutput(5, HistroryNParPerMesh, "npar_min",
                              UserHistoryOperation::min);
    }
  }
  return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::int64_t iseed(iseed_org + this->gid);
  Real b_par(b_guide * cos_a), b_perp(b_guide * sin_a);
    
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->u(IDN, k, j, i) = den;
        phydro->u(IM1, k, j, i) =
            (-v_upstrm + v_pert * (2.0 * ran2(&iseed) - 1.0)) * den;
        phydro->u(IM2, k, j, i) = v_pert * (2.0 * ran2(&iseed) - 1.0) * den;
        phydro->u(IM3, k, j, i) = v_pert * (2.0 * ran2(&iseed) - 1.0) * den;
        if (MAGNETIC_FIELDS_ENABLED) {
          if (NON_BAROTROPIC_EOS)
            phydro->u(IEN, k, j, i) = pres / (peos->GetGamma() - 1.0) +
                                      0.5 * SQR(b_guide) +
                                      0.5 *
                                          (SQR(phydro->u(IM1, k, j, i)) +
                                           SQR(phydro->u(IM2, k, j, i)) +
                                           SQR(phydro->u(IM3, k, j, i))) /
                                          phydro->u(IDN, k, j, i);
          pfield->b.x1f(k, j, i) = b_par;
          pfield->b.x2f(k, j, i) = b_perp;
          pfield->b.x3f(k, j, i) = 0.0;
          if (i == ie) pfield->b.x1f(k, j, ie + 1) = b_par;
          if (j == je) pfield->b.x2f(k, je + 1, i) = b_perp;
          if (k == ke) pfield->b.x3f(ke + 1, j, i) = 0.0;
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

  // No particles initially
  if (PARTICLES == CHARGED_PAR) {
    ppar->npar = 0;
#pragma omp critical
    {
      if (pmy_mesh->nblocal != par_add.size())
        par_add.resize(pmy_mesh->nblocal);
      if (pmy_mesh->nblocal != current_x.size())
        current_x.resize(pmy_mesh->nblocal);
      if (pmy_mesh->nblocal != num_up.size()) num_up.resize(pmy_mesh->nblocal);
    }
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AnalysisOutput(Real *spec, const int &ntbin, const
//! int &npbin, const Real &dpbin_inv, const Real &ln_p_min) const
//  \brief returns the particle spectrum, spec, with given parameter about p
//  axis (npbin, dpbin_inv, ln_p_min) and location in x direction (nxbin).

void Particles::AnalysisOutput(Real *spec_tmp, const int &nxbin_tmp, const int &npbin_tmp,
                               const Real &dpbin_inv_tmp,
                               const Real &ln_p_min_tmp) const {
  // TODO(sxc18): This code is hardcoded and could be vectorized potentially.
  const Real len_x_inv(1.0 /
                       (pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min)),
      x_min(pmy_mesh->mesh_size.x1min), v_shk(v_upstrm * ONE_3RD);
  // Loop over all the particles
  for (int k = 0; k < npar; ++k) {
    // In non-relativistic limit
    Real myp(std::sqrt(SQR(vpx(k) - v_shk) + SQR(vpy(k)) + SQR(vpz(k))));
    const int indp(static_cast<int>((std::log(myp) - ln_p_min_tmp) * dpbin_inv_tmp));
    if (indp >= npbin_tmp) continue;
    else if (indp < 0) continue;
    const int indx(static_cast<int>(nxbin_tmp * (xp(k) - x_min) * len_x_inv));
    if (indx >= nxbin_tmp) continue;
    else if (indx < 0) continue;

    ++spec_tmp[indx + indp * nxbin_tmp];
  }
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Communicate for par_add and update par_remove.
//========================================================================================

void Mesh::UserWorkInLoop() { 
  if (PARTICLES == CHARGED_PAR) {
    if (time < t_remove) {
      int *n_new;  // How many new particles (in each process)
      n_new = new int[Globals::nranks]();
      for (auto iter = par_add.begin(); iter != par_add.end(); ++iter)
        n_new[Globals::my_rank] += iter->size();
#ifdef MPI_PARALLEL
      MPI_Allreduce(MPI_IN_PLACE, n_new, Globals::nranks, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
#endif
      int n_new_global(0), n_new_index(0);
      for (int i = 0; i < Globals::nranks; ++i) n_new_global += n_new[i];
      // Shift to the beginning index in the whole array
      for (int i = 0; i < Globals::my_rank; ++i) n_new_index += n_new[i];
      delete[] n_new;

      int *init_mbid, *pid;
      init_mbid = new int[n_new_global]();
      pid = new int[n_new_global]();
      for (auto iter1 = par_add.begin(); iter1 != par_add.end(); ++iter1) {
        for (auto iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2) {
          init_mbid[n_new_index] = iter2->first;
          pid[n_new_index] = iter2->second;
          ++n_new_index;
        }
      }
#ifdef MPI_PARALLEL
      MPI_Allreduce(MPI_IN_PLACE, init_mbid, n_new_global, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, pid, n_new_global, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
#endif
      // Update par_remove
      for (int i = 0; i < n_new_global; ++i)
        par_remove.push_back(std::make_pair(init_mbid[i], pid[i]));
      delete[] init_mbid;
      delete[] pid;
      // Finally return par_add to 0
      for (auto iter = par_add.begin(); iter != par_add.end(); ++iter)
        std::vector<std::pair<int, int>>().swap(*iter);
    } else if (time > 2.0 * t_remove && analysis_enable &&
               (std::floor(time * analysis_t_inv) !=
                std::floor((time + dt) * analysis_t_inv))) {
#pragma ivdep
      std::fill(&spec[0], &spec[npbin * nxbin], 0);

      // Count the particle spectrum in the mesh
      for (int i = 0; i < this->nblocal; ++i)
        this->my_blocks(i)->ppar->AnalysisOutput(spec, nxbin, npbin, dpbin_inv,
                                                 ln_p_min);

        // sum over all ranks
#ifdef MPI_PARALLEL
      if (Globals::my_rank == 0)
        MPI_Reduce(MPI_IN_PLACE, spec, nxbin * npbin, MPI_ATHENA_REAL, MPI_SUM,
                   0, MPI_COMM_WORLD);
      else
        MPI_Reduce(spec, spec, nxbin * npbin, MPI_ATHENA_REAL, MPI_SUM, 0,
                   MPI_COMM_WORLD);
#endif  // MPI_PARALLEL
      if (Globals::my_rank == 0) {
        // normalization
        Real norm(dpbin_inv * static_cast<Real>(nxbin) /
                  (mesh_size.x1max - mesh_size.x1min));
        norm /= std::accumulate(&spec[0], &spec[nxbin * npbin], 0);
#pragma omp simd simdlen(SIMD_WIDTH)
        for (int i = 0; i < nxbin * npbin; ++i) spec[i] *= norm;
        
        // write output file
        std::string fname;
        // Construct file name
        char number[6];
        std::snprintf(number, sizeof(number), "%05d",
                      1 + static_cast<int>(std::floor(time * analysis_t_inv)));
        fname.assign(spec_basename);
        fname.append(".");
        fname.append(number);
        fname.append(".spec");

        // open file for output
        FILE *pfile;
        std::stringstream msg;
        if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
          msg << "### FATAL ERROR in function [Mesh::UserWorkInLoop]"
              << std::endl
              << "Output file '" << fname << "' could not be opened"
              << std::endl;
          ATHENA_ERROR(msg);
        }

        // print file header
        std::fprintf(pfile, "# Time:\n");
        std::fprintf(pfile, "%e\n", time);
        std::fprintf(pfile, "# p_min, p_max, nbinp:\n");
        std::fprintf(pfile, "%e %e %d\n", *(p_bin.begin()), *(p_bin.end() - 1),
                     npbin);
        std::fprintf(pfile, "# x1min, x1max, nx:\n");
        std::fprintf(pfile, "%e %e %d\n", mesh_size.x1min, mesh_size.x1max,
                     nxbin);
        std::fprintf(pfile, "\n");

        // loop over all points in spec arrays
        Real nxbin_inv(1.0 / static_cast<Real>(nxbin));
        for (int i = 0; i < nxbin * npbin; ++i) {
          int indp(static_cast<int>(i * nxbin_inv));
          int indt(i - indp * nxbin);
          std::fprintf(
              pfile, "%e %e %e\n", p_bin[indp],
              mesh_size.x1min + (indt + 0.5) * nxbin_inv *
                                    (mesh_size.x1max - mesh_size.x1min),
              spec[i]);
        }
        std::fclose(pfile);
      }
    }
  }
  if ((time - 2.0 * t_remove) *
          (time + dt - 2.0 * t_remove) <
      0.0) {
    // Clear par_remove
    std::vector<std::pair<int, int>>().swap(par_remove);
  }
  return;
}

void MeshBlock::UserWorkInLoop() {
#if (PARTICLES == CHARGED_PAR)
  // Firstly to initailize par_add, current_x, and num_up
#pragma omp critical
  {
    if (pmy_mesh->nblocal != par_add.size()) 
      par_add.resize(pmy_mesh->nblocal);
    if (pmy_mesh->nblocal != current_x.size())
      current_x.resize(pmy_mesh->nblocal);
    if (pmy_mesh->nblocal != num_up.size())
      num_up.resize(pmy_mesh->nblocal);
  }
  const Real dxi1(RINF), dxi2(pmy_mesh->f2 ? RINF : 0),
      dxi3(pmy_mesh->f3 ? RINF : 0);
  const int npc1(NPC), npc2(pmy_mesh->f2 ? NPC : 1),
      npc3(pmy_mesh->f3 ? NPC : 1);
  const Real c_reciprocal_square(1.0 /
                                 SQR(ChargedParticles::GetSpeedOfLight())),
      v_shk(v_upstrm * ONE_3RD);
  const Real x_shock(pmy_mesh->mesh_size.x1min + v_shk * pmy_mesh->time);
  const std::vector<Real> q_tmp(ChargedParticles::GetOneParticlecharge()),
      q_over_m_c(ppar->GetTypes());
  std::int64_t iseed(iseed_org + this->gid +
                     std::floor((pmy_mesh->time + pmy_mesh->dt)));
  Real begin_prim[(NHYDRO)];

  if (block_size.x1max < x_shock) {  // In the downstream
    for (int k = 0; k < ppar->npar;) {
      Real temp(SQR(ppar->vpx(k)) + SQR(ppar->vpy(k)) + SQR(ppar->vpz(k)));
      Real E_k(temp / (std::sqrt(1.0 + temp * c_reciprocal_square) + 1.0));
      if (E_k < energy_thermal) {  // Absorb particles if their energy is low
        if (Particles::BackReactionEnable()) {
          // Assign the absorbed particle's mass, momentum and energy to the gas
          // in TSC
          Real xi1(ppar->xi1(k)), xi2(ppar->xi2(k)), xi3(ppar->xi3(k));
          Real dxi(0);
          Real mass_tmp(
              q_tmp[ppar->tid(k)] /
              (q_over_m_c[ppar->tid(k)] * ChargedParticles::GetSpeedOfLight()));
          const Real mom1(ppar->vpx(k) * mass_tmp),
              mom2(ppar->vpy(k) * mass_tmp), mom3(ppar->vpz(k) * mass_tmp);
          int ix1(static_cast<int>(xi1 - dxi1)),
              ix2(static_cast<int>(xi2 - dxi2)),
              ix3(static_cast<int>(xi3 - dxi3));
          xi1 = ix1 + 0.5 - xi1;
          xi2 = ix2 + 0.5 - xi2;
          xi3 = ix3 + 0.5 - xi3;
#pragma loop count(NPC)
          for (int ipc3 = 0; ipc3 < npc3; ++ipc3) {
            dxi = std::min(std::abs(xi3 + ipc3), 1.5);
            const int ima3(ix3 + ipc3);
            const Real w3((pmy_mesh->f3 ? dxi < 0.5 ? 0.75 - SQR(dxi)
                                                    : 0.5 * SQR(1.5 - dxi)
                                        : 1.0) *
                          mass_tmp / pcoord->dx3f(ima3));
#pragma loop count(NPC)
            for (int ipc2 = 0; ipc2 < npc2; ++ipc2) {
              dxi = std::min(std::abs(xi2 + ipc2), 1.5);
              const int ima2(ix2 + ipc2);
              const Real w2((pmy_mesh->f2 ? dxi < 0.5 ? 0.75 - SQR(dxi)
                                                      : 0.5 * SQR(1.5 - dxi)
                                          : 1.0) /
                            pcoord->dx2f(ima2));
#pragma loop count(NPC)
              for (int ipc1 = 0; ipc1 < npc1; ++ipc1) {
                dxi = std::min(std::abs(xi2 + ipc2), 1.5);
                const int ima1(ix1 + ipc1);
                Real w((dxi < 0.5 ? 0.75 - SQR(dxi) : 0.5 * SQR(1.5 - dxi)) *
                       w2 * w3 / pcoord->dx1f(ima1));
                begin_prim[IDN] = phydro->w(IDN, ima3, ima2, ima1);
                begin_prim[IVX] = phydro->w(IVX, ima3, ima2, ima1);
                begin_prim[IVY] = phydro->w(IVY, ima3, ima2, ima1);
                begin_prim[IVZ] = phydro->w(IVZ, ima3, ima2, ima1);
                if (NON_BAROTROPIC_EOS)
                  begin_prim[IPR] = phydro->w(IPR, ima3, ima2, ima1);

                phydro->w(IDN, ima3, ima2, ima1) = begin_prim[IDN] + w;
                Real di(1.0 / phydro->w(IDN, ima3, ima2, ima1));
                phydro->w(IVX, ima3, ima2, ima1) =
                    di * (begin_prim[IVX] * begin_prim[IDN] + w * ppar->vpx(k));
                phydro->w(IVY, ima3, ima2, ima1) =
                    di * (begin_prim[IVY] * begin_prim[IDN] + w * ppar->vpy(k));
                phydro->w(IVZ, ima3, ima2, ima1) =
                    di * (begin_prim[IVZ] * begin_prim[IDN] + w * ppar->vpz(k));
                // Apply pressure floor
                if (NON_BAROTROPIC_EOS)
                  phydro->w(IPR, ima3, ima2, ima1) = std::max(
                    begin_prim[IPR] +
                        (peos->GetGamma() - 1.0) *
                            (0.5 *
                                 (begin_prim[IDN] * (SQR(begin_prim[IVX]) +
                                                     SQR(begin_prim[IVY]) +
                                                     SQR(begin_prim[IVZ])) -
                                  phydro->w(IDN, ima3, ima2, ima1) *
                                      (SQR(phydro->w(IVX, ima3, ima2, ima1)) +
                                       SQR(phydro->w(IVY, ima3, ima2, ima1)) +
                                       SQR(phydro->w(IVZ, ima3, ima2, ima1)))) +
                             w * E_k),
                    pressure_floor);
              }
            }
          }
        }
        // Then delete the corresponding particle
        ppar->RemoveOneParticle(k);
      } else {
        ++k;
      }
    }
  } else if (block_size.x1min < x_shock &&
             block_size.x1max >= x_shock) {  // The shock front inside
    // Absorb low energy partice in downstream
    for (int k = 0; k < ppar->npar;) {
      Real temp(SQR(ppar->vpx(k)) + SQR(ppar->vpy(k)) + SQR(ppar->vpz(k)));
      Real E_k(temp / (std::sqrt(1.0 + temp * c_reciprocal_square) + 1.0));
      // Absorb particles if their energy is low
      if (E_k < energy_thermal && ppar->xp(k) <= x_shock) {
        if (Particles::BackReactionEnable()) {
          // Assign the absorbed particle's mass, momentum and energy to the gas
          // in TSC
          Real xi1(ppar->xi1(k)), xi2(ppar->xi2(k)), xi3(ppar->xi3(k));
          Real dxi(0);
          Real mass_tmp(
              q_tmp[ppar->tid(k)] /
              (q_over_m_c[ppar->tid(k)] * ChargedParticles::GetSpeedOfLight()));
          const Real mom1(ppar->vpx(k) * mass_tmp),
              mom2(ppar->vpy(k) * mass_tmp), mom3(ppar->vpz(k) * mass_tmp);
          int ix1(static_cast<int>(xi1 - dxi1)),
              ix2(static_cast<int>(xi2 - dxi2)),
              ix3(static_cast<int>(xi3 - dxi3));
          xi1 = ix1 + 0.5 - xi1;
          xi2 = ix2 + 0.5 - xi2;
          xi3 = ix3 + 0.5 - xi3;
#pragma loop count(NPC)
          for (int ipc3 = 0; ipc3 < npc3; ++ipc3) {
            dxi = std::min(std::abs(xi3 + ipc3), 1.5);
            const int ima3(ix3 + ipc3);
            const Real w3((pmy_mesh->f3 ? dxi < 0.5 ? 0.75 - SQR(dxi)
                                                    : 0.5 * SQR(1.5 - dxi)
                                        : 1.0) *
                          mass_tmp / pcoord->dx3f(ima3));
#pragma loop count(NPC)
            for (int ipc2 = 0; ipc2 < npc2; ++ipc2) {
              dxi = std::min(std::abs(xi2 + ipc2), 1.5);
              const int ima2(ix2 + ipc2);
              const Real w2((pmy_mesh->f2 ? dxi < 0.5 ? 0.75 - SQR(dxi)
                                                      : 0.5 * SQR(1.5 - dxi)
                                          : 1.0) /
                            pcoord->dx2f(ima2));
#pragma loop count(NPC)
              for (int ipc1 = 0; ipc1 < npc1; ++ipc1) {
                dxi = std::min(std::abs(xi2 + ipc2), 1.5);
                const int ima1(ix1 + ipc1);
                Real w((dxi < 0.5 ? 0.75 - SQR(dxi) : 0.5 * SQR(1.5 - dxi)) *
                       w2 * w3 / pcoord->dx1f(ima1));
                begin_prim[IDN] = phydro->w(IDN, ima3, ima2, ima1);
                begin_prim[IVX] = phydro->w(IVX, ima3, ima2, ima1);
                begin_prim[IVY] = phydro->w(IVY, ima3, ima2, ima1);
                begin_prim[IVZ] = phydro->w(IVZ, ima3, ima2, ima1);
                if (NON_BAROTROPIC_EOS)
                  begin_prim[IPR] = phydro->w(IPR, ima3, ima2, ima1);

                phydro->w(IDN, ima3, ima2, ima1) = begin_prim[IDN] + w;
                Real di(1.0 / phydro->w(IDN, ima3, ima2, ima1));
                phydro->w(IVX, ima3, ima2, ima1) =
                    di * (begin_prim[IVX] * begin_prim[IDN] + w * ppar->vpx(k));
                phydro->w(IVY, ima3, ima2, ima1) =
                    di * (begin_prim[IVY] * begin_prim[IDN] + w * ppar->vpy(k));
                phydro->w(IVZ, ima3, ima2, ima1) =
                    di * (begin_prim[IVZ] * begin_prim[IDN] + w * ppar->vpz(k));
                // Apply pressure floor
                if (NON_BAROTROPIC_EOS)
                  phydro->w(IPR, ima3, ima2, ima1) = std::max(
                    begin_prim[IPR] +
                        (peos->GetGamma() - 1.0) *
                            (0.5 *
                                 (begin_prim[IDN] * (SQR(begin_prim[IVX]) +
                                                     SQR(begin_prim[IVY]) +
                                                     SQR(begin_prim[IVZ])) -
                                  phydro->w(IDN, ima3, ima2, ima1) *
                                      (SQR(phydro->w(IVX, ima3, ima2, ima1)) +
                                       SQR(phydro->w(IVY, ima3, ima2, ima1)) +
                                       SQR(phydro->w(IVZ, ima3, ima2, ima1)))) +
                             w * E_k),
                    pressure_floor);
              }
            }
          }
        }
        // Then delete the corresponding particle
        ppar->RemoveOneParticle(k);
      } else {
        ++k;
      }
    }

    const int npar_old(ppar->npar);
    const Real mass_inj(q_tmp[0] /
                        (q_over_m_c[0] * ChargedParticles::GetSpeedOfLight()));
    // Inject fresh particles in the downstream
    const Real u_vel(v_upstrm * std::sqrt(energy_cr_inj));
    const Real dx_fresh(v_upstrm * pmy_mesh->dt * (4.0 / 3.0)),
        r_g(u_vel / (b_guide * q_over_m_c[0]));
    Real dm_cr(0);
    int i_shock(is);
    for (int i = is; i <= ie; ++i) {
      if (pcoord->x1f(i) < x_shock && x_shock <= pcoord->x1f(i + 1)) {
        i_shock = i;
        break;
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        // Counting how many CR to inject
        Real dm_cr_column(0);   
        for (int i = is - NGHOST; i <= i_shock; ++i) {
          if ((x_shock - pcoord->x1f(i + 1)) < dx_fresh)
            dm_cr_column += phydro->w(IDN, k, j, i) *
                     std::min(pcoord->dx1f(i),
                              dx_fresh -
                                  (x_shock - pcoord->x1f(i + 1)));
        }
        dm_cr_column += phydro->w(IDN, k, j, i_shock) *
                 (x_shock - pcoord->x1f(i_shock + 1));
        dm_cr += dm_cr_column * pcoord->dx2f(j) * pcoord->dx3f(k);
      }
    }
    const int n_inj(std::round(dm_cr * cr_frac / mass_inj));
    
    ppar->npar += n_inj;
    if (ppar->npar > ppar->nparmax)
      ppar->UpdateCapacity(ppar->npar * 2 - ppar->nparmax);
    // range to inject particles
    const Real dy_inj(block_size.x2max - block_size.x2min),
        dz_inj(block_size.x3max - block_size.x3min);
    for (int n = (ppar->npar - n_inj); n < ppar->npar; ++n) {
      // For loaction in Cartesian
      ppar->xp(n) =
          x_shock -
          ran2(&iseed) * std::min(dx_fresh + r_g, x_shock - pcoord->x1f(is));
      ppar->yp(n) = block_size.x2min + ran2(&iseed) * dy_inj;
      ppar->zp(n) = block_size.x3min + ran2(&iseed) * dz_inj;

      // Boost particle velocity to the lab frame
      Real mytheta(std::acos(2.0 * ran2(&iseed) - 1.0)),
          myphi(TWO_PI * ran2(&iseed)), u1, u2, u3, denorm, mygam;
      if (mytheta < 0) mytheta += PI;
      u1 = u_vel * cos(mytheta);
      u2 = u_vel * sin(mytheta) * cos(myphi);
      u3 = u_vel * sin(mytheta) * sin(myphi);
      denorm = 1.0 + v_shk * u1 * c_reciprocal_square;
      mygam = 1.0 / std::sqrt(1.0 - SQR(v_shk) * c_reciprocal_square);
      ppar->vpx(n) = (v_shk + u1) / denorm;
      ppar->vpy(n) = u2 / (denorm * mygam);
      ppar->vpz(n) = u3 / (denorm * mygam);
      mygam = 1.0 / sqrt(1.0 - (SQR(ppar->vpx(n)) + SQR(ppar->vpy(n)) +
                                SQR(ppar->vpz(n))) *
                                   c_reciprocal_square);
      ppar->vpx(n) *= mygam;
      ppar->vpy(n) *= mygam;
      ppar->vpz(n) *= mygam;
      ppar->tid(n) = 0;  // Monotype particles injected
      // Set indices
      pcoord->CartesianToMeshCoords(ppar->xp(n), ppar->yp(n), ppar->zp(n), u1,
                                    u2, u3);
      pcoord->MeshCoordsToIndices(u1, u2, u3, ppar->xi1(n), ppar->xi2(n),
                                  ppar->xi3(n));

      // Subtract mass, energy, momentum from gas
      if (Particles::BackReactionEnable()) {
        // Assign the injected particle's mass, momentum and energy to the gas
        // in TSC
        Real xi1(ppar->xi1(n)), xi2(ppar->xi2(n)), xi3(ppar->xi3(n));
        Real dxi(0);
        Real temp(SQR(ppar->vpx(n)) + SQR(ppar->vpy(n)) + SQR(ppar->vpz(n)));
        const Real mom1(ppar->vpx(n) * mass_inj), mom2(ppar->vpy(n) * mass_inj),
            mom3(ppar->vpz(n) * mass_inj),
            energy_temp(mass_inj * temp /
                        (std::sqrt(1.0 + temp * c_reciprocal_square) + 1.0));
        int ix1(static_cast<int>(xi1 - dxi1)),
            ix2(static_cast<int>(xi2 - dxi2)),
            ix3(static_cast<int>(xi3 - dxi3));
        xi1 = ix1 + 0.5 - xi1;
        xi2 = ix2 + 0.5 - xi2;
        xi3 = ix3 + 0.5 - xi3;
#pragma loop count(NPC)
        for (int ipc3 = 0; ipc3 < npc3; ++ipc3) {
          dxi = std::min(std::abs(xi3 + ipc3), 1.5);
          const int ima3(ix3 + ipc3);
          const Real w3(
              (pmy_mesh->f3 ? dxi < 0.5 ? 0.75 - SQR(dxi) : 0.5 * SQR(1.5 - dxi)
                            : 1.0) /
              pcoord->dx3f(ima3));
#pragma loop count(NPC)
          for (int ipc2 = 0; ipc2 < npc2; ++ipc2) {
            dxi = std::min(std::abs(xi2 + ipc2), 1.5);
            const int ima2(ix2 + ipc2);
            const Real w2((pmy_mesh->f2 ? dxi < 0.5 ? 0.75 - SQR(dxi)
                                                    : 0.5 * SQR(1.5 - dxi)
                                        : 1.0) /
                          pcoord->dx2f(ima2));
#pragma loop count(NPC)
            for (int ipc1 = 0; ipc1 < npc1; ++ipc1) {
              dxi = std::min(std::abs(xi2 + ipc2), 1.5);
              const int ima1(ix1 + ipc1);
              Real w((dxi < 0.5 ? 0.75 - SQR(dxi) : 0.5 * SQR(1.5 - dxi)) * w2 *
                     w3 / pcoord->dx1f(ima1));
              begin_prim[IDN] = phydro->w(IDN, ima3, ima2, ima1);
              begin_prim[IVX] = phydro->w(IVX, ima3, ima2, ima1);
              begin_prim[IVY] = phydro->w(IVY, ima3, ima2, ima1);
              begin_prim[IVZ] = phydro->w(IVZ, ima3, ima2, ima1);
              if (NON_BAROTROPIC_EOS)
                begin_prim[IPR] = phydro->w(IPR, ima3, ima2, ima1);

              // Apply density floor
              phydro->w(IDN, ima3, ima2, ima1) =
                  std::max(begin_prim[IDN] - mass_inj * w, density_floor);
              Real di(1.0 / phydro->w(IDN, ima3, ima2, ima1));
              phydro->w(IVX, ima3, ima2, ima1) =
                  di * (begin_prim[IVX] * begin_prim[IDN] - w * mom1);
              phydro->w(IVY, ima3, ima2, ima1) =
                  di * (begin_prim[IVY] * begin_prim[IDN] - w * mom2);
              phydro->w(IVZ, ima3, ima2, ima1) =
                  di * (begin_prim[IVZ] * begin_prim[IDN] - w * mom3);

              // Apply pressure floor
              if (NON_BAROTROPIC_EOS)
                phydro->w(IPR, ima3, ima2, ima1) = std::max(
                  begin_prim[IPR] +
                      (peos->GetGamma() - 1.0) *
                          (0.5 * (begin_prim[IDN] * (SQR(begin_prim[IVX]) +
                                                     SQR(begin_prim[IVY]) +
                                                     SQR(begin_prim[IVZ])) -
                                  phydro->w(IDN, ima3, ima2, ima1) *
                                      (SQR(phydro->w(IVX, ima3, ima2, ima1)) +
                                       SQR(phydro->w(IVY, ima3, ima2, ima1)) +
                                       SQR(phydro->w(IVZ, ima3, ima2, ima1)))) -
                           w * energy_temp),
                  pressure_floor);
            }
          }
        }
      }
    }

    ppar->SetNewParticleID();
    // Record correponoding particles if t < t_remove
    if (pmy_mesh->time < t_remove) {
      for (int i = npar_old; i < ppar->npar; i++)
        par_add[lid].push_back(
            std::make_pair(ppar->init_mbid(i), ppar->pid(i)));
    }
  }

  // for all mesh block, remove particles in par_remove at 2 * t_remove
  if ((pmy_mesh->time - 2.0 * t_remove) *
          (pmy_mesh->time + pmy_mesh->dt - 2.0 * t_remove) <
      0.0) {
    for (int k = 0; k < ppar->npar;) {
      std::vector<std::pair<int, int>>::iterator iter =
          std::find(par_remove.begin(), par_remove.end(),
                    std::make_pair(ppar->init_mbid(k), ppar->pid(k)));
      if (iter != par_remove.end())
        ppar->RemoveOneParticle(k);  // This particle is injected before t_remove
      else
        ++k;
    }
  }
#endif  // (PARTICLES == CHARGED_PAR)
  return;
}

// Sum up current in x dir, stored in current_x[pmy_block->lid]
void Particles::UserWorkInLoop() {
  const Real v_shk(v_upstrm * ONE_3RD);
  const Real x_shock(pmy_mesh->mesh_size.x1min + v_shk * pmy_mesh->time);
 
  current_x[pmy_block->lid] = 0;
  num_up[pmy_block->lid] = 0;
  if (pmy_block->block_size.x1min >= x_shock) {  // In the upstream
    for (int k = 0; k < npar; ++k) current_x[pmy_block->lid] += vpx(k);
    num_up[pmy_block->lid] = npar;
  } else if (pmy_block->block_size.x1min < x_shock &&
             pmy_block->block_size.x1max >= x_shock) {  // The shock front inside
    for (int k = 0; k < npar; ++k) {  
      if (xp(k) > x_shock) {
        current_x[pmy_block->lid] += vpx(k);
        ++num_up[pmy_block->lid];
      }
    }
  }
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn int RefinementCondition(MeshBlock *pmb)
//! \brief refinement condition: maximum density and pressure curvature

int RefinementCondition(MeshBlock *pmb) {
  AthenaArray<Real> &w = pmb->phydro->w;
  Real maxeps = 0.0;
  int k = pmb->ks;
  for (int j = pmb->js; j <= pmb->je; j++) {
    for (int i = pmb->is; i <= pmb->ie; i++) {
      Real epsr = (std::abs(w(IDN, k, j, i + 1) - 2.0 * w(IDN, k, j, i) +
                            w(IDN, k, j, i - 1)) +
                   std::abs(w(IDN, k, j + 1, i) - 2.0 * w(IDN, k, j, i) +
                            w(IDN, k, j - 1, i))) /
                  w(IDN, k, j, i);
      Real epsp(0);
      if (NON_BAROTROPIC_EOS)
        epsp = (std::abs(w(IPR, k, j, i + 1) - 2.0 * w(IPR, k, j, i) +
                         w(IPR, k, j, i - 1)) +
                std::abs(w(IPR, k, j + 1, i) - 2.0 * w(IPR, k, j, i) +
                         w(IPR, k, j - 1, i))) /
               w(IPR, k, j, i);
      Real eps = std::max(epsr, epsp);
      maxeps = std::max(maxeps, eps);
    }
  }
  // refine : curvature > 1.0
  if (maxeps > 1.0) return 1;
  // derefinement: curvature < 0.1
  if (maxeps < 0.1) return -1;
  // otherwise, stay
  return 0;
}

Real HistroryCurrentXUpStream(MeshBlock *pmb, int iout) {
  if (PARTICLES != 0) pmb->ppar->UserWorkInLoop();
  if (PARTICLES == 0)
    return 0.0;
  else
  return current_x[pmb->lid];
}

Real HistroryNumUpStream(MeshBlock *pmb, int iout) {
  if (PARTICLES == 0)
    return 0.0;
  else
    return num_up[pmb->lid];
}

Real HistroryNBPerMesh(MeshBlock *pmb, int iout) {
  return static_cast<Real>(pmb->pmy_mesh->nblocal) /
         static_cast<Real>(pmb->pmy_mesh->nbtotal);
}

Real HistroryNParPerMesh(MeshBlock *pmb, int iout) { 
    int npar_total(0);
  for (int i = 0; i < pmb->pmy_mesh->nblocal; ++i) 
      npar_total += pmb->pmy_mesh->my_blocks(i)->ppar->GetLocalNumber();
  
  return npar_total;
}

}  // namespace