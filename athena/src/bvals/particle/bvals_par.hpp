#ifndef BVALS_PARTICLE_BVALS_PAR_HPP_
#define BVALS_PARTICLE_BVALS_PAR_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone!@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file bvals_par_mesh.hpp
//! \brief handle boundaries for Particles class

// C headers

// C++ headers
#include <array>
#include <cstring>    // std::memcpy

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../bvals_interfaces.hpp"

// MPI headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \class CellCenteredBoundaryVariable
//! \brief

class ParBoundaryVariable : public BoundaryVariable {
 public:
  ParBoundaryVariable(MeshBlock *pmb, AthenaArray<int> *intprop,
                      AthenaArray<Real> *realprop, AthenaArray<Real> *auxprop);
  ~ParBoundaryVariable();

  //! \note
  static void Initialize(const int &nint_tmp, const int &nreal_tmp, const int &naux_tmp);
  
  inline int BufidFromMeshIndex(const double &x1i, const double &x2i,
                                const double &x3i) const {
    // No 0.5 here, referring to Coordinates::MeshCoordsToIndices
    // Only double can ensure the accuracy
    return bufid_
        [static_cast<int>((x3i - pmy_block_->ks) * reciprocal_size_3_ + 1.0)]
        [static_cast<int>((x2i - pmy_block_->js) * reciprocal_size_2_ + 1.0)]
        [static_cast<int>((x1i - pmy_block_->is) * reciprocal_size_1_ + 1.0)];
    // Do not use std::ceil or std::floor, which will slow down everything
  };
  int LoadPar(const int &k, const int &bufid, const bool &pre);
  void AllocateSend(int par_size, const int &bufid) {
    par_size *= unit_size;
    if (par_size > send_size[bufid] * sizeof(Real)) {
      send_size[bufid] =
          static_cast<int>(static_cast<float>(par_size) * 2.0 / sizeof(Real) +
                           1.0) -
          send_size[bufid];
      delete[] bd_var_.send[bufid];
      bd_var_.send[bufid] = new Real[send_size[bufid]];
    }
  };
  //! As there are no flux and shear box, the only transfer value is particle
  static constexpr int max_phys_id = 2;

  //!@{
  //! BoundaryVariable:
  int ComputeVariableBufferSize(const NeighborIndexes &ni, int cng) override;
  int ComputeFluxCorrectionBufferSize(const NeighborIndexes &ni,
                                      int cng) override;
  //!@}

  //!@{
  //! BoundaryCommunication:
  void SetupPersistentMPI() override;
  void StartReceiving(BoundaryCommSubset phase) override;
  void ClearBoundary(BoundaryCommSubset phase) override;
  void StartReceivingShear(BoundaryCommSubset phase) override;
  //!@}

  //!@{
  //! BoundaryBuffer:
  void SendBoundaryBuffers() override;
  bool ReceiveBoundaryBuffers() override;
  void ReceiveAndSetBoundariesWithWait() override;
  void SetBoundaries() override;
  void SendFluxCorrection() override;
  bool ReceiveFluxCorrection() override;
  //!@}

  //!@{
  //! BoundaryPhysics:
  void ReflectInnerX1(Real time, Real dt, int il, int jl, int ju, int kl,
                      int ku, int ngh) override;
  void ReflectOuterX1(Real time, Real dt, int iu, int jl, int ju, int kl,
                      int ku, int ngh) override;
  void ReflectInnerX2(Real time, Real dt, int il, int iu, int jl, int kl,
                      int ku, int ngh) override;
  void ReflectOuterX2(Real time, Real dt, int il, int iu, int ju, int kl,
                      int ku, int ngh) override;
  void ReflectInnerX3(Real time, Real dt, int il, int iu, int jl, int ju,
                      int kl, int ngh) override;
  void ReflectOuterX3(Real time, Real dt, int il, int iu, int jl, int ju,
                      int ku, int ngh) override;

  void OutflowInnerX1(Real time, Real dt, int il, int jl, int ju, int kl,
                      int ku, int ngh) override;
  void OutflowOuterX1(Real time, Real dt, int iu, int jl, int ju, int kl,
                      int ku, int ngh) override;
  void OutflowInnerX2(Real time, Real dt, int il, int iu, int jl, int kl,
                      int ku, int ngh) override;
  void OutflowOuterX2(Real time, Real dt, int il, int iu, int ju, int kl,
                      int ku, int ngh) override;
  void OutflowInnerX3(Real time, Real dt, int il, int iu, int jl, int ju,
                      int kl, int ngh) override;
  void OutflowOuterX3(Real time, Real dt, int il, int iu, int jl, int ju,
                      int ku, int ngh) override;

  void PolarWedgeInnerX2(Real time, Real dt, int il, int iu, int jl, int kl,
                         int ku, int ngh) override;
  void PolarWedgeOuterX2(Real time, Real dt, int il, int iu, int ju, int kl,
                         int ku, int ngh) override;
  //!@}

 protected:
  AthenaArray<int> *var_int;     //   ptr for Particles::intprop
  AthenaArray<Real> *var_real;   //   ptr for Particles::realprop
  AthenaArray<Real> *var_aux;    //   ptr for Particles::auxprop

  static int nint, nreal, naux;  // copy from Particles
  static int unit_size;          // nint_ * sizeof(int) + (nreal_ + naux_) * sizeof(Real)

  // number of particles to send and recieve (in unit of byte)
  int send_npar[56], recv_npar[56];
  // size of BoundaryData::send[i] and BoundaryData::recv[i] (in unit of sizeof(Real))
  int send_size[56], recv_size[56];
  // tag for MPI_Isend and MPI_Irecv, initialized at the beginning
  int send_tag[56], recv_tag[56];

  //!@{
  //! BoundaryVariable:
  void CopyVariableBufferSameProcess(NeighborBlock &nb, int ssize) override;
  //!@}
 private:
  //!@{
  //! BoundaryBuffer:
  int LoadBoundaryBufferSameLevel(Real *buf, const NeighborBlock &nb) override;
  void SetBoundarySameLevel(Real *buf, const NeighborBlock &nb) override;

  int LoadBoundaryBufferToCoarser(Real *buf, const NeighborBlock &nb) override;
  int LoadBoundaryBufferToFiner(Real *buf, const NeighborBlock &nb) override;
  //!@}

  void SetBoundaryFromCoarser(Real *buf, const NeighborBlock &nb) override;
  void SetBoundaryFromFiner(Real *buf, const NeighborBlock &nb) override;

  void PolarBoundarySingleAzimuthalBlock() override;

  void ReAllocateSend(const int &ssize_byte, const int &bufid) {
    if (ssize_byte > send_size[bufid] * sizeof(Real)) {
      int send_size_new(
          static_cast<int>(static_cast<float>(ssize_byte) * 2.0 / sizeof(Real) +
                           1.0) -
          send_size[bufid]);
      Real *buffer_new = new Real[send_size_new];
#pragma ivdep
      std::memcpy(buffer_new, bd_var_.send[bufid], ssize_byte - unit_size);
      send_size[bufid] = send_size_new;
      delete[] bd_var_.send[bufid];
      bd_var_.send[bufid] = buffer_new;
    }
  };
  void AllocateRecv(const int &rsize_byte, const int &bufid) {
    if (rsize_byte > recv_size[bufid] * sizeof(Real)) {
      recv_size[bufid] =
          static_cast<int>(static_cast<float>(rsize_byte) * 2.0 / sizeof(Real) +
                           1.0) -
          recv_size[bufid];
      // Do not need to copy the existing bd_var_.recv[bufid]
      delete[] bd_var_.recv[bufid];
      bd_var_.recv[bufid] = new Real[recv_size[bufid]];
    }
  };
  void InitializeBufid();

  // Working variables
#ifdef MPI_PARALLEL
  int par_phys_id_;
#endif

  double reciprocal_size_1_, reciprocal_size_2_, reciprocal_size_3_;
  int bufid_[4][4][4]; // Initialized with -2 (null neighbours)
};

#endif  // BVALS_PARTICLE_BVALS_PAR_HPP_
