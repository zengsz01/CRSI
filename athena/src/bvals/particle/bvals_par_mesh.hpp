#ifndef BVALS_PARTICLE_BVALS_PAR_MESH_HPP_
#define BVALS_PARTICLE_BVALS_PAR_MESH_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone!@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file bvals_par_mesh.hpp
//! \brief handle boundaries for ParticleMesh class

// C headers

// C++ headers
#include <array>

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../bvals.hpp"
#include "../bvals_interfaces.hpp"

// MPI headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \class CellCenteredBoundaryVariable
//! \brief

class ParMeshBoundaryVariable : public BoundaryVariable {
 public:
  ParMeshBoundaryVariable(MeshBlock *pmb, AthenaArray<Real> *var);
  ~ParMeshBoundaryVariable();

  //! \note
  std::array<AthenaArray<Real>, 56>
      diff_level_var;  // may be vaccume if refinement is unsupported

  //! As there are no flux and shear box, the only transfer value is particle feedback
  static constexpr int max_phys_id = 1;

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
  
  // Unique public function
  void InitDiffLevelVar();
 protected:
  int nl_, nu_;
  // Even though "do_nothing_" only controls functions in the derived class, but
  // Particles::backreaction can control whether to call functions in the task list

 private:
  AthenaArray<Real> *var_meshaux_;  // ptr for ParticleMesh::meshaux, in mesh coord
  
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

#ifdef MPI_PARALLEL
  int pm_phys_id_;
#endif
  // TODO(sxc18): Implementation for the shearing box
};

#endif  // BVALS_PARTICLE_BVALS_PAR_MESH_HPP_
