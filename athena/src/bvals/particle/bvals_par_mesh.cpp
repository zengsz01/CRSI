//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file bvals_par_mesh.cpp
//! \brief functions that apply BCs for ParticleMesh class

// C headers

// C++ headers
#include <cstdlib>
#include <cstring>    // memcpy()
#include <string>     // c_str()
#include <cmath>      // std::max(), std::min()

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../globals.hpp"
#include "../../mesh/mesh.hpp"
#include "../../particles/particles.hpp"
#include "../../parameter_input.hpp"
#include "../../utils/buffer_utils.hpp"
#include "../bvals.hpp"
#include "bvals_par_mesh.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//! constructor
//! 
ParMeshBoundaryVariable::ParMeshBoundaryVariable(MeshBlock* pmb,
                                                   AthenaArray<Real>* var)
    : BoundaryVariable(pmb),
      var_meshaux_(var),
      nl_(0),
      nu_(var->GetDim4() - 1) {

  // same as BoundaryQuantity::cc
  InitBoundaryData(bd_var_, BoundaryQuantity::cc);
#ifdef MPI_PARALLEL
  // KGF: dead code, leaving for now:
  // cc_phys_id_ = pbval_->ReserveTagVariableIDs(1);
  pm_phys_id_ = pbval_->bvars_next_phys_id_;
#endif
  // TODO(sxc18): implementation for the shearing box
}

//! destructor
//! 
ParMeshBoundaryVariable::~ParMeshBoundaryVariable() {
  DestroyBoundaryData(bd_var_);
  for (auto iter = diff_level_var.begin(); iter != diff_level_var.end(); ++iter)
    iter->DeleteAthenaArray();
}

//----------------------------------------------------------------------------------------
//! \fn int ParMeshBoundaryVariable::ComputeVariableBufferSize( const
//! NeighborIndexes& ni, int cng)
//! \brief return the max number of cells during boundary communication

int ParMeshBoundaryVariable::ComputeVariableBufferSize(
    const NeighborIndexes& ni, int cng) {
  const RegionSize& block_size = pmy_block_->block_size;
  // to the same level
  int size(((ni.ox1 == 0) ? block_size.nx1 : NGHOST) *
                ((ni.ox2 == 0) ? block_size.nx2 : NGHOST) *
                ((ni.ox3 == 0) ? block_size.nx3 : NGHOST));
  if (pmy_mesh_->multilevel) {
    const int NGH(cng - 1), NG2(2 * NGHOST);
    const int nx1h((pmy_block_->pmy_mesh->mesh_size.nx1 > 1)
                       ? block_size.nx1 / 2 + NGH
                       : 1),
        nx2h((pmy_block_->pmy_mesh->mesh_size.nx2 > 1)
                 ? block_size.nx2 / 2 + NGH
                 : 1),
        nx3h((pmy_block_->pmy_mesh->mesh_size.nx3 > 1)
                 ? block_size.nx3 / 2 + NGH
                 : 1);
    int f2c(((ni.ox1 == 0) ? nx1h : NGH) * ((ni.ox2 == 0) ? nx2h : NGH) *
            ((ni.ox3 == 0) ? nx3h : NGH)),
        c2f(((ni.ox1 == 0) ? block_size.nx1 : NG2) *
            ((ni.ox2 == 0) ? block_size.nx2 : NG2) *
            ((ni.ox3 == 0) ? block_size.nx3 : NG2));
    size = std::max(size, c2f);
    size = std::max(size, f2c);
  }
  return (nu_ + 1) * size;
}

//----------------------------------------------------------------------------------------
//! \fn int ParMeshBoundaryVariable::ComputeFluxCorrectionBufferSize(const
//! NeighborIndexes& ni, int cng)
//! \brief there is no flux correction, is return 0

int ParMeshBoundaryVariable::ComputeFluxCorrectionBufferSize(
    const NeighborIndexes& ni, int cng) {
  // As there may be segmentation fault when new Real[0], default size is 1
  return 1;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::SetupPersistentMPI()
//! \brief Setup persistent MPI requests to be reused throughout the entire
//! simulation

void ParMeshBoundaryVariable::SetupPersistentMPI() {
  MeshBlock* pmb = pmy_block_;
  
  const RegionSize& block_size = pmy_block_->block_size;
  const int mylevel = pmb->loc.level;
  const int NGH = pmb->cnghost - 1, NG2 = 2 * NGHOST;
  const int nx1h(
      (pmy_block_->pmy_mesh->mesh_size.nx1 > 1) ? block_size.nx1 / 2 + NGH : 1),
      nx2h((pmy_block_->pmy_mesh->mesh_size.nx2 > 1) ? block_size.nx2 / 2 + NGH
                                                     : 1),
      nx3h((pmy_block_->pmy_mesh->mesh_size.nx3 > 1) ? block_size.nx3 / 2 + NGH
                                                     : 1);
#ifdef MPI_PARALLEL
  int ssize, rsize;
  int tag;
  // Initialize non-polar neighbor communications to other ranks
  for (int n = 0; n < pbval_->nneighbor; ++n) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      if (nb.snb.level == mylevel) {  // same
        ssize = rsize = ((nb.ni.ox1 == 0) ? block_size.nx1 : NGHOST) *
                        ((nb.ni.ox2 == 0) ? block_size.nx2 : NGHOST) *
                        ((nb.ni.ox3 == 0) ? block_size.nx3 : NGHOST);
      } else if (nb.snb.level < mylevel) {  // coarser
        ssize = ((nb.ni.ox1 == 0) ? nx1h : NGH) *
                ((nb.ni.ox2 == 0) ? nx2h : NGH) *
                ((nb.ni.ox3 == 0) ? nx3h : NGH);
        rsize = ((nb.ni.ox1 == 0) ? block_size.nx1 : NG2) *
                ((nb.ni.ox2 == 0) ? block_size.nx2 : NG2) *
                ((nb.ni.ox3 == 0) ? block_size.nx3 : NG2);
      } else {  // finer
        ssize = ((nb.ni.ox1 == 0) ? block_size.nx1 : NG2) *
                ((nb.ni.ox2 == 0) ? block_size.nx2 : NG2) *
                ((nb.ni.ox3 == 0) ? block_size.nx3 : NG2);
        rsize = ((nb.ni.ox1 == 0) ? nx1h : NGH) *
                ((nb.ni.ox2 == 0) ? nx2h : NGH) *
                ((nb.ni.ox3 == 0) ? nx3h : NGH);
      }
      ssize *= (nu_ + 1);
      rsize *= (nu_ + 1);
      // specify the offsets in the view point of the target block: flip ox?
      // signs

      // Initialize persistent communication requests attached to specific
      // BoundaryData cell-centered hydro: bd_hydro_
      tag = pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid, pm_phys_id_);
      if (bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_send[nb.bufid]);
      MPI_Send_init(bd_var_.send[nb.bufid], ssize, MPI_ATHENA_REAL, nb.snb.rank,
                    tag, MPI_COMM_WORLD, &(bd_var_.req_send[nb.bufid]));
      tag = pbval_->CreateBvalsMPITag(pmb->lid, nb.bufid, pm_phys_id_);
      if (bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_recv[nb.bufid]);
      MPI_Recv_init(bd_var_.recv[nb.bufid], rsize, MPI_ATHENA_REAL, nb.snb.rank,
                    tag, MPI_COMM_WORLD, &(bd_var_.req_recv[nb.bufid]));
    }
  }
#endif

  // Initilize diff_level_var (destroy in ~ParMeshBoundaryVariable())
  if (pmy_mesh_->multilevel) {
    for (auto iter = diff_level_var.begin(); iter != diff_level_var.end(); ++iter)
      iter->DeleteAthenaArray();

    // Initiate ParticleMesh boundary data.
    pmb->ppar->SetParMeshBoundaryAttributes();

    for (int n = 0; n < pbval_->nneighbor; ++n) {
      NeighborBlock& nb = pbval_->neighbor[n];
      
      if (nb.snb.level < mylevel)  // coarser
        diff_level_var[nb.bufid].NewAthenaArray(
            nu_ + 1, (nb.ni.ox3 == 0) ? nx3h : NGH,
            (nb.ni.ox2 == 0) ? nx2h : NGH, (nb.ni.ox1 == 0) ? nx1h : NGH);
      else if (nb.snb.level > mylevel)  // finer
        diff_level_var[nb.bufid].NewAthenaArray(
            nu_ + 1, (nb.ni.ox3 == 0) ? block_size.nx3 : NG2,
            (nb.ni.ox2 == 0) ? block_size.nx2 : NG2,
            (nb.ni.ox1 == 0) ? block_size.nx1 : NG2);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::StartReceiving(BoundaryCommSubset phase)
//! \brief initiate MPI_Irecv()

void ParMeshBoundaryVariable::StartReceiving(BoundaryCommSubset phase) {
  // Temparily do not support orbital advection
  if (phase == BoundaryCommSubset::orbital) return;
#ifdef MPI_PARALLEL
  for (int n = 0; n < pbval_->nneighbor; ++n) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank)
      MPI_Start(&(bd_var_.req_recv[nb.bufid]));
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::ClearBoundary(BoundaryCommSubset phase)
//! \brief clean up the boundary flags after each loop

void ParMeshBoundaryVariable::ClearBoundary(BoundaryCommSubset phase) {
  // Temparily do not support orbital advection
  if (phase == BoundaryCommSubset::orbital) return;
  for (int n = 0; n < pbval_->nneighbor; ++n) {
    NeighborBlock& nb = pbval_->neighbor[n];
    bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
    bd_var_.sflag[nb.bufid] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
    if (nb.snb.rank != Globals::my_rank) {
      // Wait for Isend
      MPI_Wait(&(bd_var_.req_send[nb.bufid]), MPI_STATUS_IGNORE);
    }
#endif
  }
  return;
}

void ParMeshBoundaryVariable::StartReceivingShear(BoundaryCommSubset phase) {
  return;
}

void ParMeshBoundaryVariable::SendFluxCorrection() { return; }

bool ParMeshBoundaryVariable::ReceiveFluxCorrection() { return true; }

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::ReflectInnerX1(Real time, Real dt, int
//! il, int jl, int ju, int kl, int ku, int ngh)
//! \brief REFLECTING boundary conditions, inner x1 boundary

void ParMeshBoundaryVariable::ReflectInnerX1(Real time, Real dt, int il,
                                              int jl, int ju, int kl, int ku,
                                              int ngh) {
  AthenaArray<Real>& var = *var_meshaux_;
  for (int n = 0; n <= nu_; ++n) {
    const Real sign((n == Particles::Index_X_Vel()) ? -1.0 : 1.0);
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
#pragma omp simd
        for (int i = 1; i <= ngh; ++i) {
          var(n, k, j, (il + i - 1)) += sign * var(n, k, j, il - i);
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::ReflectOuterX1(Real time, Real dt, int
//! iu, int jl, int ju, int kl, int ku, int ngh)
//! \brief REFLECTING boundary conditions, outer x1 boundary

void ParMeshBoundaryVariable::ReflectOuterX1(Real time, Real dt, int iu,
                                              int jl, int ju, int kl, int ku,
                                              int ngh) {
  AthenaArray<Real>& var = *var_meshaux_;
  for (int n = 0; n <= nu_; ++n) {
    const Real sign((n == Particles::Index_X_Vel()) ? -1.0 : 1.0);
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
#pragma omp simd
        for (int i = 1; i <= ngh; ++i) {
          var(n, k, j, (iu - i + 1)) += sign * var(n, k, j, iu + i);
        }
      }
    }
  }
  return;

}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::ReflectInnerX2(Real time, Real dt, int
//! il, int iu, int jl, int kl, int ku, int ngh)
//! \brief REFLECTING boundary conditions, inner x2 boundary

void ParMeshBoundaryVariable::ReflectInnerX2(Real time, Real dt, int il,
                                              int iu, int jl, int kl, int ku,
                                              int ngh) {
  AthenaArray<Real>& var = *var_meshaux_;
  for (int n = 0; n <= nu_; ++n) {
    const Real sign((n == Particles::Index_Y_Vel()) ? -1.0 : 1.0);
    for (int k = kl; k <= ku; ++k) {
      for (int j = 1; j <= ngh; ++j) {
#pragma omp simd
        for (int i = il; i <= iu; ++i) {
          var(n, k, jl + j - 1, i) += sign * var(n, k, jl - j, i);
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::ReflectOuterX2(Real time, Real dt, int
//! il, int iu, int ju, int kl, int ku, int ngh)
//! \brief REFLECTING boundary conditions, outer x2 boundary

void ParMeshBoundaryVariable::ReflectOuterX2(Real time, Real dt, int il,
                                              int iu, int ju, int kl, int ku,
                                              int ngh) {
  AthenaArray<Real>& var = *var_meshaux_;
  for (int n = 0; n <= nu_; ++n) {
    const Real sign((n == Particles::Index_Y_Vel()) ? -1.0 : 1.0);
    for (int k = kl; k <= ku; ++k) {
      for (int j = 1; j <= ngh; ++j) {
#pragma omp simd
        for (int i = il; i <= iu; ++i) {
          var(n, k, ju - j + 1, i) += sign * var(n, k, ju + j, i);
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::ReflectInnerX3(Real time, Real dt, int
//! il, int iu, int jl, int ju, int kl, int ngh)
//! \brief REFLECTING boundary conditions, inner x3 boundary

void ParMeshBoundaryVariable::ReflectInnerX3(Real time, Real dt, int il,
                                              int iu, int jl, int ju, int kl,
                                              int ngh) {
  AthenaArray<Real>& var = *var_meshaux_;
  for (int n = 0; n <= nu_; ++n) {
    const Real sign((n == Particles::Index_Z_Vel()) ? -1.0 : 1.0);
    for (int k = 1; k <= ngh; ++k) {
      for (int j = jl; j <= ju; ++j) {
#pragma omp simd
        for (int i = il; i <= iu; ++i) {
          var(n, kl + k - 1, j, i) += sign * var(n, kl - k, j, i);
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::ReflectOuterX3(Real time, Real dt, int
//! il, int iu, int jl, int ju, int ku, int ngh)
//! \brief REFLECTING boundary conditions, outer x3 boundary

void ParMeshBoundaryVariable::ReflectOuterX3(Real time, Real dt, int il,
                                              int iu, int jl, int ju, int ku,
                                              int ngh) {
  AthenaArray<Real>& var = *var_meshaux_;
  for (int n = 0; n <= nu_; ++n) {
    const Real sign((n == Particles::Index_Z_Vel()) ? -1.0 : 1.0);
    for (int k = 1; k <= ngh; ++k) {
      for (int j = jl; j <= ju; ++j) {
#pragma omp simd
        for (int i = il; i <= iu; ++i) {
          var(n, ku - k + 1, j, i) += sign * var(n, ku + k, j, i);
        }
      }
    }
  }
  return;
}

void ParMeshBoundaryVariable::OutflowInnerX1(Real time, Real dt, int il,
                                              int jl, int ju, int kl, int ku,
                                              int ngh) {
  // Do nothing for the cells located close to the outflow region.
  return;
}

void ParMeshBoundaryVariable::OutflowOuterX1(Real time, Real dt, int iu,
                                              int jl, int ju, int kl, int ku,
                                              int ngh) {
  return;
}

void ParMeshBoundaryVariable::OutflowInnerX2(Real time, Real dt, int il,
                                              int iu, int jl, int kl, int ku,
                                              int ngh) {
  return;
}

void ParMeshBoundaryVariable::OutflowOuterX2(Real time, Real dt, int il,
                                              int iu, int ju, int kl, int ku,
                                              int ngh) {
  return;
}

void ParMeshBoundaryVariable::OutflowInnerX3(Real time, Real dt, int il,
                                              int iu, int jl, int ju, int kl,
                                              int ngh) {
  return;
}

void ParMeshBoundaryVariable::OutflowOuterX3(Real time, Real dt, int il,
                                              int iu, int jl, int ju, int ku,
                                              int ngh) {
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::PolarWedgeInnerX2(Real time, Real dt, int
//! il, int iu, int jl, int kl, int ku, int ngh)
//! \brief polar wedge boundary conditions, inner x2 boundary

void ParMeshBoundaryVariable::PolarWedgeInnerX2(Real time, Real dt, int il,
                                                 int iu, int jl, int kl, int ku,
                                                 int ngh) {
  AthenaArray<Real>& var = *var_meshaux_;
  for (int n = 0; n <= nu_; ++n) {
    // Rotating particles, with the feedback as well
    Real sign((n == Particles::Index_Y_Vel() || n == Particles::Index_Z_Vel())
                  ? -1.0
                  : 1.0);
    for (int k = kl; k <= ku; ++k) {
      for (int j = 1; j <= ngh; ++j) {
#pragma omp simd
        for (int i = il; i <= iu; ++i) {
          var(n, k, jl + j - 1, i) += sign * var(n, k, jl - j, i);
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::PolarWedgeOuterX2(Real time, Real dt, int
//! il, int iu, int ju, int kl, int ku, int ngh)
//! \brief polar wedge boundary conditions, outer x2 boundary

void ParMeshBoundaryVariable::PolarWedgeOuterX2(Real time, Real dt, int il,
                                                 int iu, int ju, int kl, int ku,
                                                 int ngh) {
  AthenaArray<Real>& var = *var_meshaux_;
  for (int n = 0; n <= nu_; ++n) {
    // Rotating particles, with the feedback as well
    Real sign((n == Particles::Index_Y_Vel() || n == Particles::Index_Z_Vel())
                  ? -1.0
                  : 1.0);
    for (int k = kl; k <= ku; ++k) {
      for (int j = 1; j <= ngh; ++j) {
#pragma omp simd
        for (int i = il; i <= iu; ++i) {
          var(n, k, ju - j + 1, i) += sign * var(n, k, ju + j, i);
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::InitDiffLevelVar()
//! \brief set all values in diff_level_var to 0

void ParMeshBoundaryVariable::InitDiffLevelVar() {
  if (!pmy_mesh_->multilevel) return;

  const int mylevel = pmy_block_->loc.level;
  for (int n = 0; n < pbval_->nneighbor; ++n) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.level != mylevel) {
#pragma ivdep
      std::fill(&diff_level_var[nb.bufid](0, 0, 0, 0),
                &diff_level_var[nb.bufid](nu_ + 1, 0, 0, 0), 0.0);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn int ParMeshBoundaryVariable::LoadBoundaryBufferSameLevel(Real* buf,
//! const NeighborBlock& nb)
//! \brief Set particle_mesh boundary buffers for sending to a block on the same
//! level. Package all the data even though dpe (energy changes, the last
//! dimension) is not used in the predictor step

int ParMeshBoundaryVariable::LoadBoundaryBufferSameLevel(
    Real* buf, const NeighborBlock& nb) {
  // Find the index domain.
  int iss = pmy_block_->is, ise = pmy_block_->ie, jss = pmy_block_->js,
      jse = pmy_block_->je, kss = pmy_block_->ks, kse = pmy_block_->ke;
  
  if (nb.ni.ox1 > 0) {
    iss = pmy_block_->ie + 1;
    ise += NGHOST;
  } else if (nb.ni.ox1 < 0) {
    iss -= NGHOST;
    ise = pmy_block_->is - 1;
  }

  if (nb.ni.ox2 > 0) {
    jss = pmy_block_->je + 1;
    jse += NGHOST;
  } else if (nb.ni.ox2 < 0) {
    jss -= NGHOST;
    jse = pmy_block_->js - 1;
  }

  if (nb.ni.ox3 > 0) {
    kss = pmy_block_->ke + 1;
    kse += NGHOST;
  } else if (nb.ni.ox3 < 0) {
    kss -= NGHOST;
    kse = pmy_block_->ks - 1;
  }

  int p = 0;
  AthenaArray<Real>& var = *var_meshaux_;
  BufferUtility::PackData(var, buf, nl_, nu_, iss, ise, jss, jse, kss, kse, p);
  return p;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::SetBoundarySameLevel(Real* buf, const
//! NeighborBlock& nb)
//! \brief Set particle_mesh boundary received from a block on the same level

void ParMeshBoundaryVariable::SetBoundarySameLevel(Real* buf,
                                                    const NeighborBlock& nb) {
  MeshBlock* pmb = pmy_block_;
  int irs(pmb->is), jrs(pmb->js), krs(pmb->ks), ire(pmb->ie), jre(pmb->je),
      kre(pmb->ke);
  AthenaArray<Real>& var = *var_meshaux_;
  const NeighborIndexes ni = nb.ni;

  if (ni.ox1 > 0)
    irs = pmb->ie - NGHOST + 1;
  else if (ni.ox1 < 0)
    ire = pmb->is + NGHOST - 1;

  if (ni.ox2 > 0)
    jrs = pmb->je - NGHOST + 1;
  else if (ni.ox2 < 0)
    jre = pmb->js + NGHOST - 1;

  if (ni.ox3 > 0)
    krs = pmb->ke - NGHOST + 1;
  else if (ni.ox3 < 0)
    kre = pmb->ks + NGHOST - 1;
  

  int p = 0;
  if (nb.polar) {
    const int nx3_half = (pmb->ke - pmb->ks + 1) / 2;
    for (int n = nl_; n <= nu_; ++n) {
      Real sign((n == Particles::Index_Y_Vel() || n == Particles::Index_Z_Vel())
                    ? -1.0
                    : 1.0);
      for (int k = krs; k <= kre; ++k) {
        for (int j = jre; j >= jrs; --j) {
          if (pmb->loc.level == pmy_mesh_->root_level &&
            pmy_mesh_->nrbx3 == 1 && pmb->block_size.nx3 > 1 && ni.ox2 != 0) {
            // Taking care of single MeshBlock spans the entire azimuthal
            int k_shift(k + (k < (nx3_half + NGHOST) ? 1 : -1) * nx3_half);
#pragma omp simd linear(p)
            for (int i = irs; i <= ire; ++i) {
              var(n, k_shift, j, i) += sign * buf[p++];
            }
          } else {
#pragma omp simd linear(p)
            for (int i = irs; i <= ire; ++i) {
              var(n, k, j, i) += sign * buf[p++];
            }
          }
        }
      }
    }
  } else {
    BufferUtility::UnpackAndAddData(buf, var, nl_, nu_, irs, ire, jrs, jre, krs, kre,
                                    p);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn int ParMeshBoundaryVariable::LoadBoundaryBufferToCoarser(Real* buf,
//! const NeighborBlock& nb)
//! \brief Set particle_mesh boundary buffers for sending to a block on the coarser level

int ParMeshBoundaryVariable::LoadBoundaryBufferToCoarser(
    Real* buf, const NeighborBlock& nb) {
  int p = 0;
  BufferUtility::PackData(diff_level_var[nb.bufid], buf, nl_, nu_, 0,
                          diff_level_var[nb.bufid].GetDim1() - 1, 0,
                          diff_level_var[nb.bufid].GetDim2() - 1, 0,
                          diff_level_var[nb.bufid].GetDim3() - 1, p);
  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int ParMeshBoundaryVariable::LoadBoundaryBufferToFiner(Real* buf,
//! const NeighborBlock& nb)
//! \brief Set particle_mesh boundary buffers for sending to a block on the finer level

int ParMeshBoundaryVariable::LoadBoundaryBufferToFiner(
    Real* buf, const NeighborBlock& nb) {
  int p = 0;
  BufferUtility::PackData(diff_level_var[nb.bufid], buf, nl_, nu_, 0,
                          diff_level_var[nb.bufid].GetDim1() - 1, 0,
                          diff_level_var[nb.bufid].GetDim2() - 1, 0,
                          diff_level_var[nb.bufid].GetDim3() - 1, p);
  return p;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::SetBoundaryFromCoarser(Real* buf, const
//! NeighborBlock& nb)
//! \brief Set particle_mesh buffer received from a block on a coarser level

void ParMeshBoundaryVariable::SetBoundaryFromCoarser(Real* buf,
                                                      const NeighborBlock& nb) {
  MeshBlock* pmb = pmy_block_;
  int irs(pmb->is), jrs(pmb->js), krs(pmb->ks), ire(pmb->ie), jre(pmb->je),
      kre(pmb->ke);
  AthenaArray<Real>& var = *var_meshaux_;
  const NeighborIndexes ni = nb.ni;
  const int dxir(2 * NGHOST);

  if (ni.ox1 > 0)
    irs = pmb->ie - dxir + 1;
  else if (ni.ox1 < 0)
    ire = pmb->is + dxir - 1;

  if (ni.ox2 > 0)
    jrs = pmb->je - dxir + 1;
  else if (ni.ox2 < 0)
    jre = pmb->js + dxir - 1;

  if (ni.ox3 > 0)
    krs = pmb->ke - dxir + 1;
  else if (ni.ox3 < 0)
    kre = pmb->ks + dxir - 1;

  int p = 0;
  if (nb.polar) {
    for (int n = nl_; n <= nu_; ++n) {
      Real sign((n == Particles::Index_Y_Vel() || n == Particles::Index_Z_Vel())
                    ? -1.0
                    : 1.0);
      for (int k = krs; k <= kre; ++k) {
        for (int j = jre; j >= jrs; --j) {
#pragma omp simd linear(p)
          for (int i = irs; i <= ire; ++i) {
            var(n, k, j, i) += sign * buf[p++];
          }
        }
      }
    }
  } else {
    BufferUtility::UnpackAndAddData(buf, var, nl_, nu_, irs, ire, jrs, jre, krs,
                                    kre, p);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::SetBoundaryFromFiner(Real* buf, const
//! NeighborBlock& nb)
//! \brief Set particle_mesh buffer received from a block on a finer level

void ParMeshBoundaryVariable::SetBoundaryFromFiner(Real* buf,
                                                    const NeighborBlock& nb) {
  MeshBlock* pmb = pmy_block_;
  int irs(pmb->is), jrs(pmb->js), krs(pmb->ks), ire(pmb->ie), jre(pmb->je),
      kre(pmb->ke);
  AthenaArray<Real>& var = *var_meshaux_;
  const NeighborIndexes ni = nb.ni;
  const int dxir(pmb->cnghost - 1);
  const bool active1(pmb->pmy_mesh->mesh_size.nx1 > 1),
      active2(pmb->pmy_mesh->mesh_size.nx2 > 1),
      active3(pmb->pmy_mesh->mesh_size.nx3 > 1);
  const int nx1h = active1 ? pmb->block_size.nx1 / 2 + dxir : 1,
            nx2h = active2 ? pmb->block_size.nx2 / 2 + dxir : 1,
            nx3h = active3 ? pmb->block_size.nx3 / 2 + dxir : 1;

  if (ni.ox1 > 0)
    irs = pmb->ie - dxir + 1;
  else if (ni.ox1 < 0)
    ire = pmb->is + dxir - 1;

  if (ni.ox2 > 0)
    jrs = pmb->je - dxir + 1;
  else if (ni.ox2 < 0)
    jre = pmb->js + dxir - 1;

  if (ni.ox3 > 0)
    krs = pmb->ke - dxir + 1;
  else if (ni.ox3 < 0)
    kre = pmb->ks + dxir - 1;

  if (ni.type == NeighborConnect::face) {
    if (ni.ox1 != 0) {
      if (active2) {
        if (ni.fi1) {
          jrs = pmb->je - nx2h + 1;
        } else {
          jre = pmb->js + nx2h - 1;
        }
      }
      if (active3) {
        if (ni.fi2) {
          krs = pmb->ke - nx3h + 1;
        } else {
          kre = pmb->ks + nx3h - 1;
        }
      }
    } else if (ni.ox2 != 0) {
      if (active1) {
        if (ni.fi1) {
          irs = pmb->ie - nx1h + 1;
        } else {
          ire = pmb->is + nx1h - 1;
        }
      }
      if (active3) {
        if (ni.fi2) {
          krs = pmb->ke - nx3h + 1;
        } else {
          kre = pmb->ks + nx3h - 1;
        }
      }
    } else {
      if (active1) {
        if (ni.fi1) {
          irs = pmb->ie - nx1h + 1;
        } else {
          ire = pmb->is + nx1h - 1;
        }
      }
      if (active2) {
        if (ni.fi2) {
          jrs = pmb->je - nx2h + 1;
        } else {
          jre = pmb->js + nx2h - 1;
        }
      }
    }
  } else if (ni.type == NeighborConnect::edge) {
    if (ni.ox1 == 0) {
      if (active1) {
        if (ni.fi1) {
          irs = pmb->ie - nx1h + 1;
        } else {
          ire = pmb->is + nx1h - 1;
        }
      }
    } else if (ni.ox2 == 0) {
      if (active2) {
        if (ni.fi1) {
          jrs = pmb->je - nx2h + 1;
        } else {
          jre = pmb->js + nx2h - 1;
        }
      }
    } else {
      if (active3) {
        if (ni.fi1) {
          krs = pmb->ke - nx3h + 1;
        } else {
          kre = pmb->ks + nx3h - 1;
        }
      }
    }
  }

  int p = 0;
  if (nb.polar) {
    for (int n = nl_; n <= nu_; ++n) {
      Real sign((n == Particles::Index_Y_Vel() || n == Particles::Index_Z_Vel())
                    ? -1.0
                    : 1.0);
      for (int k = krs; k <= kre; ++k) {
        for (int j = jre; j >= jrs; --j) {
#pragma omp simd linear(p)
          for (int i = irs; i <= ire; ++i) {
            var(n, k, j, i) += sign * buf[p++];
          }
        }
      }
    }
  } else {
    BufferUtility::UnpackAndAddData(buf, var, nl_, nu_, irs, ire, jrs, jre, krs,
                                    kre, p);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParMeshBoundaryVariable::PolarBoundarySingleAzimuthalBlock()
//! \brief polar boundary edge-case: done in ParMeshBoundaryVariable::SetBoundarySameLevel

void ParMeshBoundaryVariable::PolarBoundarySingleAzimuthalBlock() { return; }
