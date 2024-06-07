//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file bvals_par.cpp
//! \brief functions that apply BCs for Particles class

#define ONCE  // TWICE or ONCE

// C headers

// C++ headers
#include <cstring>    // std::memcpy
#include <numeric>    // std::accumulate
#include <cmath>      // std::ceil
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <iostream>   // std::endl

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../globals.hpp"
#include "../../mesh/mesh.hpp"
#include "../../particles/particles.hpp"
#include "../bvals.hpp"
#include "../bvals_interfaces.hpp"
#include "bvals_par.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace {
// Packaging one particle's information
#ifdef MPI_PARALLEL
MPI_Datatype MPI_ATHENA_PAR(MPI_DATATYPE_NULL);
#endif
}

// Class variable initialization
int ParBoundaryVariable::nint = 0, ParBoundaryVariable::nreal = 0,
    ParBoundaryVariable::naux = 0;
int ParBoundaryVariable::unit_size = 0;

//! constructor
//!
ParBoundaryVariable::ParBoundaryVariable(MeshBlock *pmb,
                                         AthenaArray<int> *intprop,
                                         AthenaArray<Real> *realprop,
                                         AthenaArray<Real> *auxprop)
    : BoundaryVariable(pmb),
      var_int(intprop),
      var_real(realprop),
      var_aux(auxprop),
      reciprocal_size_1_(2.0 / pmb->block_size.nx1),
      reciprocal_size_2_(2.0 / pmb->block_size.nx2),
      reciprocal_size_3_(2.0 / pmb->block_size.nx3) {
  InitBoundaryData(bd_var_, BoundaryQuantity::par);
#pragma omp simd simdlen(SIMD_WIDTH)
  for (int n = 0; n < bd_var_.nbmax; ++n) {
    send_npar[n] = 0;
    recv_npar[n] = 0;
    // corresponding to InitBoundaryData(bd_var_, BoundaryQuantity::par);
    send_size[n] = 1;
    recv_size[n] = 1;
    send_tag[n] = -1;
    recv_tag[n] = -1;
  }

#ifdef MPI_PARALLEL
  // KGF: dead code, leaving for now:
  // cc_phys_id_ = pbval_->ReserveTagVariableIDs(1);
  par_phys_id_ = pbval_->bvars_next_phys_id_;
#endif
}

//! destructor
//!
ParBoundaryVariable::~ParBoundaryVariable() { DestroyBoundaryData(bd_var_); }

//--------------------------------------------------------------------------------------
//! \fn void ParBoundaryVariable::Initialize(const int &nint_tmp, const int
//! &nreal_tmp, const int &naux_tmp)
//! \brief initializes the class.

void ParBoundaryVariable::Initialize(const int &nint_tmp, const int &nreal_tmp,
                                     const int &naux_tmp) {
  nint = nint_tmp;
  nreal = nreal_tmp;
  naux = naux_tmp;
  unit_size = nint * sizeof(int) + (nreal + naux) * sizeof(Real);
#ifdef MPI_PARALLEL
  if (MPI_DATATYPE_NULL != MPI_ATHENA_PAR) MPI_Type_free(&MPI_ATHENA_PAR);
  MPI_Type_contiguous(unit_size, MPI_BYTE, &MPI_ATHENA_PAR);
  MPI_Type_commit(&MPI_ATHENA_PAR);
  // Caution: Nowhere to call MPI_Type_free(&MPI_ATHENA_PAR)
#endif
}

//----------------------------------------------------------------------------------------
//! \fn ParBoundaryVariable::LoadPar(const int &k, const int &bufid)
//! \brief package particle k into bd_var_.send[bufid] and return
//! send_npar[bufid]. pre indicate whether to ReAllocateSend 

int ParBoundaryVariable::LoadPar(const int &k, const int &bufid,
                                 const bool &pre) {
  /*if (bufid < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ParBoundaryVariable::LoadPar" << std::endl
        << "bufid = " << bufid << " was passed\n"
        << "Should be bufid >= 0." << std::endl;
    ATHENA_ERROR(msg);
  }*/
  ++send_npar[bufid];
  if (!pre) ReAllocateSend(send_npar[bufid] * unit_size, bufid);
  // Caution: If pre == true, the following part may overflow
  // Caution: slightly unsafe (especially if sizeof(int) > sizeof(Real))
  char *buf_tmp = reinterpret_cast<char *> (bd_var_.send[bufid]);
  buf_tmp += (send_npar[bufid] - 1) * unit_size;  // bd_var_.send[bufid]
  int *ptr_int;          // Package int properties first
  ptr_int = reinterpret_cast<int *>(buf_tmp);
  for (int i = 0; i < nint; ++i) *(ptr_int++) = (*var_int)(i, k);
  Real *ptr_real;  // Package Real properties secondly
  ptr_real = reinterpret_cast<Real *>(ptr_int);
  for (int i = 0; i < nreal; ++i) *(ptr_real++) = (*var_real)(i, k);
  // Package auxiliary properties finally
  for (int i = 0; i < naux; ++i) *(ptr_real++) = (*var_aux)(i, k);

  return send_npar[bufid];
}

int ParBoundaryVariable::ComputeVariableBufferSize(const NeighborIndexes &ni,
                                                   int cng) {
  return 1;
}

int ParBoundaryVariable::ComputeFluxCorrectionBufferSize(
    const NeighborIndexes &ni, int cng) {
  return 1;
}

//----------------------------------------------------------------------------------------
//! \fn void ParBoundaryVariable::SendBoundaryBuffers()
//! \brief Send boundary buffers of variables, using MPI_ISend

void ParBoundaryVariable::SendBoundaryBuffers() {
  MeshBlock *pmb = pmy_block_;
  for (int n = 0; n < pbval_->nneighbor; n++) {
    NeighborBlock &nb = pbval_->neighbor[n];
    if (bd_var_.sflag[nb.bufid] == BoundaryStatus::completed) continue;
    // Do not distinguish the levels, all the data has been packaged
    if (nb.snb.rank == Globals::my_rank)  // on the same process
      CopyVariableBufferSameProcess(nb, send_npar[nb.bufid]);
#ifdef MPI_PARALLEL
    else { // MPI
      MPI_Isend(bd_var_.send[nb.bufid], send_npar[nb.bufid], MPI_ATHENA_PAR,
                nb.snb.rank, send_tag[nb.bufid], MPI_COMM_WORLD,
                &(bd_var_.req_send[nb.bufid]));
    }
#endif
    bd_var_.sflag[nb.bufid] = BoundaryStatus::completed;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParBoundaryVariable::ReceiveBoundaryBuffers()
//! \brief probe the buffer size and then receive the boundary data

bool ParBoundaryVariable::ReceiveBoundaryBuffers() {
  bool bflag = true;

  for (int n = 0; n < pbval_->nneighbor; n++) {
    NeighborBlock &nb = pbval_->neighbor[n];
    if (bd_var_.flag[nb.bufid] == BoundaryStatus::arrived ||
        (bd_var_.flag[nb.bufid] == BoundaryStatus::completed &&
         nb.snb.rank == Globals::my_rank))
      continue;
    else if (bd_var_.flag[nb.bufid] == BoundaryStatus::disabled) {
      bflag = false;
      continue;
    }
#ifdef MPI_PARALLEL
    // NOLINT // MPI boundary
    else if (bd_var_.flag[nb.bufid] == BoundaryStatus::waiting ||
             (bd_var_.flag[nb.bufid] == BoundaryStatus::completed &&
              nb.snb.rank != Globals::my_rank)) {
      int test;
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test,
                 MPI_STATUS_IGNORE);
      MPI_Test(&(bd_var_.req_recv[nb.bufid]), &test, MPI_STATUS_IGNORE);
      if (!static_cast<bool>(test)) {
        bflag = false;
        continue;
      }
      bd_var_.flag[nb.bufid] = BoundaryStatus::arrived;
    }
#endif
  }
  return bflag;
}

//----------------------------------------------------------------------------------------
//! \fn void ParBoundaryVariable::ReceiveAndSetBoundariesWithWait()
//! \brief receive and set the boundary data for initialization

void ParBoundaryVariable::ReceiveAndSetBoundariesWithWait() {
  MeshBlock *pmb = pmy_block_;
  for (int n = 0; n < pbval_->nneighbor; n++) {
    NeighborBlock &nb = pbval_->neighbor[n];
#ifdef MPI_PARALLEL
    if (nb.snb.rank != Globals::my_rank) {
      MPI_Status status;
      MPI_Probe(nb.snb.rank, recv_tag[nb.bufid], MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_ATHENA_PAR, &recv_npar[nb.bufid]);
      AllocateRecv(recv_npar[nb.bufid] * unit_size, nb.bufid);
      MPI_Recv(bd_var_.recv[nb.bufid], recv_npar[nb.bufid], MPI_ATHENA_PAR,
               nb.snb.rank, recv_tag[nb.bufid], MPI_COMM_WORLD, &status);
    }
#endif
    bd_var_.flag[nb.bufid] = BoundaryStatus::completed;  // completed
  }
  // Check the memory size
  const int sum_recv_par(
      std::accumulate(&recv_npar[0], &recv_npar[bd_var_.nbmax], 0));  // sum all
  if (sum_recv_par + pmy_block_->ppar->npar > pmy_block_->ppar->nparmax)
    pmy_block_->ppar->UpdateCapacity(2 * pmy_block_->ppar->npar +
                                     2 * sum_recv_par -
                                     pmy_block_->ppar->nparmax);
  for (int n = 0; n < pbval_->nneighbor; ++n)
    if (recv_npar[pbval_->neighbor[n].bufid] > 0)
      SetBoundarySameLevel(bd_var_.recv[pbval_->neighbor[n].bufid],
                           pbval_->neighbor[n]);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParBoundaryVariable::SetBoundaries()
//! \brief set the boundary data

void ParBoundaryVariable::SetBoundaries() {
  // Check the memory size
  const int sum_recv_par(
      std::accumulate(&recv_npar[0], &recv_npar[bd_var_.nbmax], 0));  // sum all
  if (sum_recv_par + pmy_block_->ppar->npar > pmy_block_->ppar->nparmax)
    pmy_block_->ppar->UpdateCapacity(2 * pmy_block_->ppar->npar +
                                     2 * sum_recv_par -
                                     pmy_block_->ppar->nparmax);
  for (int n = 0; n < pbval_->nneighbor; ++n) {
    NeighborBlock &nb = pbval_->neighbor[n];
    // Do not distinguish the levels, using the same callee
    // SetBoundarySameLevel()
    if (bd_var_.flag[nb.bufid] == BoundaryStatus::arrived) {
      SetBoundarySameLevel(bd_var_.recv[nb.bufid], nb);
      bd_var_.flag[nb.bufid] = BoundaryStatus::completed;  // completed
    }
  }
}

void ParBoundaryVariable::SendFluxCorrection() { return; }

//----------------------------------------------------------------------------------------
//! \fn void ParBoundaryVariable::ReceiveFluxCorrection()
//! \brief receive the number of particles (to be received) and start receiving

bool ParBoundaryVariable::ReceiveFluxCorrection() {
  bool bflag = true;

  for (int n = 0; n < pbval_->nneighbor; n++) {
    NeighborBlock &nb = pbval_->neighbor[n];
    if (bd_var_.flag[nb.bufid] == BoundaryStatus::waiting ||
        bd_var_.flag[nb.bufid] == BoundaryStatus::completed)
      continue;
    else if (bd_var_.flag[nb.bufid] == BoundaryStatus::disabled) {
      if (nb.snb.rank == Globals::my_rank) {  // on the same process
        bflag = false;
        continue;
      }
#ifdef MPI_PARALLEL
      else {  // NOLINT // MPI boundary
        int test;

        MPI_Status status;
        MPI_Iprobe(nb.snb.rank, recv_tag[nb.bufid], MPI_COMM_WORLD, &test,
                   &status);
        if (!static_cast<bool>(test)) {
          bflag = false;
          continue;
        }
        MPI_Get_count(&status, MPI_ATHENA_PAR, &recv_npar[nb.bufid]);
        if (recv_npar[nb.bufid] > 0) {
          // Before MPI_Irecv
          AllocateRecv(recv_npar[nb.bufid] * unit_size, nb.bufid);
          bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
        } else {
          bd_var_.flag[nb.bufid] = BoundaryStatus::completed;
        }
        MPI_Irecv(bd_var_.recv[nb.bufid], recv_npar[nb.bufid], MPI_ATHENA_PAR,
                  nb.snb.rank, recv_tag[nb.bufid], MPI_COMM_WORLD,
                  &(bd_var_.req_recv[nb.bufid]));
      }
#endif
    }
  }
  return bflag;
}

void ParBoundaryVariable::ReflectInnerX1(Real time, Real dt, int il, int jl,
                                         int ju, int kl, int ku, int ngh) {
  return;
}

void ParBoundaryVariable::ReflectOuterX1(Real time, Real dt, int iu, int jl,
                                         int ju, int kl, int ku, int ngh) {
  return;
}

void ParBoundaryVariable::ReflectInnerX2(Real time, Real dt, int il, int iu,
                                         int jl, int kl, int ku, int ngh) {
  return;
}

void ParBoundaryVariable::ReflectOuterX2(Real time, Real dt, int il, int iu,
                                         int ju, int kl, int ku, int ngh) {
  return;
}

void ParBoundaryVariable::ReflectInnerX3(Real time, Real dt, int il, int iu,
                                         int jl, int ju, int kl, int ngh) {
  return;
}

void ParBoundaryVariable::ReflectOuterX3(Real time, Real dt, int il, int iu,
                                         int jl, int ju, int ku, int ngh) {
  return;
}

void ParBoundaryVariable::OutflowInnerX1(Real time, Real dt, int il, int jl,
                                         int ju, int kl, int ku, int ngh) {
  return;
}

void ParBoundaryVariable::OutflowOuterX1(Real time, Real dt, int iu, int jl,
                                         int ju, int kl, int ku, int ngh) {
  return;
}

void ParBoundaryVariable::OutflowInnerX2(Real time, Real dt, int il, int iu,
                                         int jl, int kl, int ku, int ngh) {
  return;
}

void ParBoundaryVariable::OutflowOuterX2(Real time, Real dt, int il, int iu,
                                         int ju, int kl, int ku, int ngh) {
  return;
}

void ParBoundaryVariable::OutflowInnerX3(Real time, Real dt, int il, int iu,
                                         int jl, int ju, int kl, int ngh) {
  return;
}

void ParBoundaryVariable::OutflowOuterX3(Real time, Real dt, int il, int iu,
                                         int jl, int ju, int ku, int ngh) {
  return;
}

void ParBoundaryVariable::PolarWedgeInnerX2(Real time, Real dt, int il, int iu,
                                            int jl, int kl, int ku, int ngh) {
  return;
}

void ParBoundaryVariable::PolarWedgeOuterX2(Real time, Real dt, int il, int iu,
                                            int ju, int kl, int ku, int ngh) {
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParBoundaryVariable::SetupPersistentMPI()
//! \brief ParBoundaryVariable uses non-blocking but non-persisitent
//! communication, as the length of communcation array is dynamic. This function
//! only initializes tags.

void ParBoundaryVariable::SetupPersistentMPI() {
  InitializeBufid();
#ifdef MPI_PARALLEL
#pragma omp simd simdlen(SIMD_WIDTH)
  for (int n = 0; n < bd_var_.nbmax; n++) {
    send_tag[n] = -1;
    recv_tag[n] = -1;
  }
  int tag;
  for (int n = 0; n < pbval_->nneighbor; ++n) {
    NeighborBlock &nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      send_tag[nb.bufid] =
          pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid, par_phys_id_);
      recv_tag[nb.bufid] =
          pbval_->CreateBvalsMPITag(pmy_block_->lid, nb.bufid, par_phys_id_);
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void ParBoundaryVariable::StartReceiving(BoundaryCommSubset phase)
//! \brief intialize send_npar[n] and recv_npar[n] at the beginning for each cycle

void ParBoundaryVariable::StartReceiving(BoundaryCommSubset phase) {
  // Temparily do not support orbital advection
  if (phase == BoundaryCommSubset::orbital) return;
  // Set all npar = 0, not only for those used
  for (int n = 0; n < bd_var_.nbmax; n++) {
    send_npar[n] = 0;
    recv_npar[n] = 0;
    bd_var_.flag[n] = BoundaryStatus::disabled;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParBoundaryVariable::ClearBoundary(BoundaryCommSubset phase)
//! \brief clean up the boundary flags after each loop

void ParBoundaryVariable::ClearBoundary(BoundaryCommSubset phase) {
  if (phase == BoundaryCommSubset::orbital) return;
  for (int n = 0; n < pbval_->nneighbor; ++n) {
    NeighborBlock &nb = pbval_->neighbor[n];
    bd_var_.flag[nb.bufid] = BoundaryStatus::disabled;
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

void ParBoundaryVariable::StartReceivingShear(BoundaryCommSubset phase) {
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParBoundaryVariable::CopyVariableBufferSameProcess(NeighborBlock& nb,
//! int ssize) 
//! \brief Called in ParBoundaryVariable::SendBoundaryBuffer()
//! when the destination neighbor block is on the same MPI rank as the sending
//! MeshBlcok. So std::memcpy() call requires a pointer to  "void *dst"
//! corresponding to bd_var_.recv[nb.targetid] on the target block. ssize is the
//! number of particles. If ssize exceeds recv_size[nb.targetid] * sizeof(Real),
//! re-allocate recv[nb.targetid]

void ParBoundaryVariable::CopyVariableBufferSameProcess(NeighborBlock &nb,
                                                        int ssize) {
  // Locate target buffer
  // 1) which MeshBlock?
  MeshBlock *ptarget_block = pmy_mesh_->FindMeshBlock(nb.snb.gid);
  // 2) which element in vector of BoundaryVariable *?
  ParBoundaryVariable *ppbvar_tgt = static_cast<ParBoundaryVariable *>(
      ptarget_block->pbval->bvars[bvar_index]);
  ppbvar_tgt->recv_npar[nb.targetid] = ssize;
  BoundaryData<> *ptarget_bdata = &(ppbvar_tgt->bd_var_);
  if (ssize == 0) {
    ptarget_bdata->flag[nb.targetid] = BoundaryStatus::completed;
    return;
  }
  ssize *= unit_size;  // Now in unit of byte
  // 3) check the size of receiving array
  ppbvar_tgt->AllocateRecv(ssize, nb.targetid);
  std::memcpy(ptarget_bdata->recv[nb.targetid], bd_var_.send[nb.bufid], ssize);
  
  // finally, set the BoundaryStatus flag on the destination buffer
  ptarget_bdata->flag[nb.targetid] = BoundaryStatus::arrived;
  return;
}

int ParBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
                                                     const NeighborBlock &nb) {
  return 1;
}

//----------------------------------------------------------------------------------------
//! \fn void ParBoundaryVariable::SetBoundarySameLevel(Real* buf, const
//! NeighborBlock& nb)
//! \brief Set particle boundary received from neighbours, no matter which level
//! neighbour locates

void ParBoundaryVariable::SetBoundarySameLevel(Real *buf,
                                               const NeighborBlock &nb) {
  char *buf_tmp = (char *)buf;
#pragma omp simd simdlen(SIMD_WIDTH)
  for (int k = pmy_block_->ppar->npar;
       k < pmy_block_->ppar->npar + recv_npar[nb.bufid]; ++k) {
    int *ptr_int;  // Unpack int properties first
    // Caution: slightly unsafe (especially if sizeof(int) > sizeof(Real))
    ptr_int = reinterpret_cast<int *>(buf_tmp +
                                      (k - pmy_block_->ppar->npar) * unit_size);
    for (int i = 0; i < nint; ++i) (*var_int)(i, k) = *(ptr_int++);
    Real *ptr_real;  // Unpack Real properties secondly
    ptr_real = reinterpret_cast<Real *> (ptr_int);
    for (int i = 0; i < nreal; ++i) (*var_real)(i, k) = *(ptr_real++);
    // Unpack auxiliary properties finally
    for (int i = 0; i < naux; ++i) (*var_aux)(i, k) = *(ptr_real++);
  }
  pmy_block_->ppar->npar += recv_npar[nb.bufid];
}

int ParBoundaryVariable::LoadBoundaryBufferToCoarser(Real *buf,
                                                     const NeighborBlock &nb) {
  return 1;
}

int ParBoundaryVariable::LoadBoundaryBufferToFiner(Real *buf,
                                                   const NeighborBlock &nb) {
  return 1;
}

void ParBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
                                                 const NeighborBlock &nb) {
  return;
}

void ParBoundaryVariable::SetBoundaryFromFiner(Real *buf,
                                               const NeighborBlock &nb) {
  return;
}

void ParBoundaryVariable::PolarBoundarySingleAzimuthalBlock() { return; }

//----------------------------------------------------------------------------------------
//! \fn void ParBoundaryVariable::InitializeBufid()
//! \brief Initializing bufid_. -2 refers to non-block, -1 refers to the meshblock itself.

void ParBoundaryVariable::InitializeBufid() {
  // Initializing all as null
  for (int k = 0; k < 4; ++k)
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i) bufid_[k][j][i] = -2;

  // TODO(sxc18): The following calculation may not be the best. It is copied
  // from bvals_cc.cpp, bvals_refine.cpp, and mesh_refinement.cpp
  const int mylevel = pmy_block_->loc.level;
  const double cng(pmy_block_->cnghost - 1);
  for (int n = 0; n < pbval_->nneighbor; ++n) {
    NeighborBlock &nb = pbval_->neighbor[n];
    double iss, ise, jss, jse, kss, kse;

    // Find several depths from the neighbor block.
    const int nblevel = nb.snb.level;
    const NeighborIndexes &ni = nb.ni;
    if (nblevel < mylevel) {  // Neighbor block is at a coarser level.
      iss = pmy_block_->cis;
      ise = pmy_block_->cie;
      jss = pmy_block_->cjs;
      jse = pmy_block_->cje;
      kss = pmy_block_->cks;
      kse = pmy_block_->cke;
      if (nb.ni.ox1 == 0) {
        if ((pmy_block_->loc.lx1 & 1LL) == 0LL)
          ise += cng;
        else
          iss -= cng;
      } else if (nb.ni.ox1 > 0) {
        iss = pmy_block_->cie + 1, ise += cng;
      } else {
        iss -= cng, ise = pmy_block_->cis - 1;
      }
      if (nb.ni.ox2 == 0) {
        if (pmy_block_->block_size.nx2 > 1) {
          if ((pmy_block_->loc.lx2 & 1LL) == 0LL)
            jse += cng;
          else
            jss -= cng;
        }
      } else if (nb.ni.ox2 > 0) {
        jss = pmy_block_->cje + 1, jse += cng;
      } else {
        jss -= cng, jse = pmy_block_->cjs - 1;
      }
      if (nb.ni.ox3 == 0) {
        if (pmy_block_->block_size.nx3 > 1) {
          if ((pmy_block_->loc.lx3 & 1LL) == 0LL)
            kse += cng;
          else
            kss -= cng;
        }
      } else if (nb.ni.ox3 > 0) {
        kss = pmy_block_->cke + 1, kse += cng;
      } else {
        kss -= cng, kse = pmy_block_->cks - 1;
      }

      if (pmy_block_->block_size.nx3 > 1) {  // 3D
        kss = (kss - pmy_block_->cks) * 2 + pmy_block_->ks;
        kse = (kse - pmy_block_->cks) * 2 + pmy_block_->ks + 1;
        jss = (jss - pmy_block_->cjs) * 2 + pmy_block_->js;
        jse = (jse - pmy_block_->cjs) * 2 + pmy_block_->js + 1;
        iss = (iss - pmy_block_->cis) * 2 + pmy_block_->is;
        ise = (ise - pmy_block_->cis) * 2 + pmy_block_->is + 1;
      } else if (pmy_block_->block_size.nx2 > 1) {  // 2D
        kss = pmy_block_->ks;
        kse = pmy_block_->ks;
        jss = (jss - pmy_block_->cjs) * 2 + pmy_block_->js;
        jse = (jse - pmy_block_->cjs) * 2 + pmy_block_->js + 1;
        iss = (iss - pmy_block_->cis) * 2 + pmy_block_->is;
        ise = (ise - pmy_block_->cis) * 2 + pmy_block_->is + 1;
      } else {  // 1D
        kss = pmy_block_->ks;
        kse = pmy_block_->ks;
        jss = pmy_block_->js;
        jse = pmy_block_->js;
        iss = (iss - pmy_block_->cis) * 2 + pmy_block_->is;
        ise = (ise - pmy_block_->cis) * 2 + pmy_block_->is + 1;
      }
    } else if (nblevel > mylevel) {  // Neighbor block is at a finer level.
      iss = pmy_block_->is;
      ise = pmy_block_->ie;
      jss = pmy_block_->js;
      jse = pmy_block_->je;
      kss = pmy_block_->ks;
      kse = pmy_block_->ke;
      if (nb.ni.ox1 == 0) {
        if (nb.ni.fi1 == 1)
          iss += pmy_block_->block_size.nx1 / 2;
        else
          ise -= pmy_block_->block_size.nx1 / 2;
      } else if (nb.ni.ox1 > 0) {
        iss = pmy_block_->ie + 1, ise += NGHOST;
      } else {
        iss -= NGHOST, ise = pmy_block_->is - 1;
      }
      if (nb.ni.ox2 == 0) {
        if (pmy_block_->block_size.nx2 > 1) {
          if (nb.ni.ox1 != 0) {
            if (nb.ni.fi1 == 1)
              jss += pmy_block_->block_size.nx2 / 2;
            else
              jse -= pmy_block_->block_size.nx2 / 2;
          } else {
            if (nb.ni.fi2 == 1)
              jss += pmy_block_->block_size.nx2 / 2;
            else
              jse -= pmy_block_->block_size.nx2 / 2;
          }
        }
      } else if (nb.ni.ox2 > 0) {
        jss = pmy_block_->je + 1, jse += NGHOST;
      } else {
        jss -= NGHOST, jse = pmy_block_->js - 1;
      }
      if (nb.ni.ox3 == 0) {
        if (pmy_block_->block_size.nx3 > 1) {
          if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
            if (nb.ni.fi1 == 1)
              kss += pmy_block_->block_size.nx3 / 2;
            else
              kse -= pmy_block_->block_size.nx3 / 2;
          } else {
            if (nb.ni.fi2 == 1)
              kss += pmy_block_->block_size.nx3 / 2;
            else
              kse -= pmy_block_->block_size.nx3 / 2;
          }
        }
      } else if (nb.ni.ox3 > 0) {
        kss = pmy_block_->ke + 1, kse += NGHOST;
      } else {
        kss -= NGHOST, kse = pmy_block_->ks - 1;
      }
    } else {  // Neighbor block is at the same level.
      iss = pmy_block_->is;
      ise = pmy_block_->ie;
      jss = pmy_block_->js;
      jse = pmy_block_->je;
      kss = pmy_block_->ks;
      kse = pmy_block_->ke;
      if (ni.ox1 > 0) {
        iss = pmy_block_->ie + 1;
        ise += NGHOST;
      } else if (ni.ox1 < 0) {
        iss -= NGHOST;
        ise = pmy_block_->is - 1;
      }

      if (ni.ox2 > 0) {
        jss = pmy_block_->je + 1;
        jse += NGHOST;
      } else if (ni.ox2 < 0) {
        jss -= NGHOST;
        jse = pmy_block_->js - 1;
      }

      if (ni.ox3 > 0) {
        kss = pmy_block_->ke + 1;
        kse += NGHOST;
      } else if (ni.ox3 < 0) {
        kss -= NGHOST;
        kse = pmy_block_->ks - 1;
      }
    }

    // Loop all possibile points in bufid_, where they maybe repeat
    for (int k = std::ceil((kss - pmy_block_->ks + 0.5) * reciprocal_size_3_);
         k <= std::ceil((kse - pmy_block_->ks + 0.5) * reciprocal_size_3_); ++k)
      for (int j = std::ceil((jss - pmy_block_->js + 0.5) * reciprocal_size_2_);
           j <= std::ceil((jse - pmy_block_->js + 0.5) * reciprocal_size_2_); ++j)
        for (int i = std::ceil((iss - pmy_block_->is + 0.5) * reciprocal_size_1_);
             i <= std::ceil((ise - pmy_block_->is + 0.5) * reciprocal_size_1_);
             ++i)
          bufid_[k][j][i] = nb.bufid;
  }

  // Then loop all points inside the meshblock itself, with bufid = -1
  for (int k = 1; k < 3; ++k)
    for (int j = 1; j < 3; ++j)
      for (int i = 1; i < 3; ++i) bufid_[k][j][i] = -1;

  // Finally loop the inactive domain
  if (!pmy_mesh_->f2) {        // 1D
    for (int i = 0; i < 4; ++i) {
      bufid_[0][0][i] = bufid_[1][1][i];    // No refinement in x2 and x3
      bufid_[0][2][i] = bufid_[1][1][i];
      bufid_[0][3][i] = bufid_[1][1][i];
      bufid_[2][0][i] = bufid_[1][1][i];
      bufid_[2][2][i] = bufid_[1][1][i];
      bufid_[2][3][i] = bufid_[1][1][i];
      bufid_[3][0][i] = bufid_[1][1][i];
      bufid_[3][2][i] = bufid_[1][1][i];
      bufid_[3][3][i] = bufid_[1][1][i];
    }
  } else if (!pmy_mesh_->f3) {  // 2D
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i) {
        bufid_[0][j][i] = bufid_[1][j][i];  // No refinement in x3
        bufid_[2][j][i] = bufid_[1][j][i];
        bufid_[3][j][i] = bufid_[1][j][i];
      }
  }
  return;
}
