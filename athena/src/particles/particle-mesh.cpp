//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//======================================================================================
//! \file particle-mesh.cpp
//  \brief implements ParticleMesh class used for operations involved in
//  particle-mesh
//         methods.

// C++ headers
#include <algorithm>  // min()
#include <cstdlib>    // posix_memalign()
#include <cstring>
#include <iostream>   // std::endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error

// Athena++ classes headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../utils/buffer_utils.hpp"
#include "../bvals/particle/bvals_par_mesh.hpp"
#include "particle-mesh.hpp"
#include "particles.hpp"

// Class variable initialization
bool ParticleMesh::initialized_ = false;
int ParticleMesh::nmeshaux = 0;
int ParticleMesh::iweight = -1;
bool ParticleMesh::active1_ = false, ParticleMesh::active2_ = false,
     ParticleMesh::active3_ = false;
Real ParticleMesh::dxi1_ = 0.0, ParticleMesh::dxi2_ = 0.0,
     ParticleMesh::dxi3_ = 0.0;
int ParticleMesh::npc1_ = 1, ParticleMesh::npc2_ = 1, ParticleMesh::npc3_ = 1;
int ParticleMesh::ncloud_ = 1, ParticleMesh::ncloud_slice_ = 1;
int ParticleMesh::ipc1v_[NPC_CUBE] = {};
int ParticleMesh::ipc2v_[NPC_CUBE] = {};
int ParticleMesh::ipc3v_[NPC_CUBE] = {};
int ParticleMesh::islice_[NPC_CUBE] = {};
int ParticleMesh::ipc2v_slice_[NPC_SQR] = {};
int ParticleMesh::ipc3v_slice_[NPC_SQR] = {};

// Local function prototypes.
namespace {
constexpr Real boundary_at_weightFunction = 1.5;
inline Real _WeightFunction(Real dxi);
}  // namespace

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::Initialize(Mesh *pm, ParameterInput *pin)
//  \brief initiates the ParticleMesh class.

void ParticleMesh::Initialize(Mesh* pm, ParameterInput* pin) {
  if (initialized_) return;

  // Determine active dimensions.
  RegionSize& mesh_size = pm->mesh_size;
  active1_ = mesh_size.nx1 > 1;
  active2_ = mesh_size.nx2 > 1;
  active3_ = mesh_size.nx3 > 1;

  // Determine the range of a particle cloud.
  dxi1_ = active1_ ? RINF : 0;
  dxi2_ = active2_ ? RINF : 0;
  dxi3_ = active3_ ? RINF : 0;

  // Determine the dimensions of each particle cloud.
  npc1_ = active1_ ? NPC : 1;
  npc2_ = active2_ ? NPC : 1;
  npc3_ = active3_ ? NPC : 1;

  ncloud_ = npc3_ * npc2_ * npc1_;
  ncloud_slice_ = npc2_ * npc3_;

  // Pre-calculate the ipc(ipc_slice) and code the transposition to 1d loop
  int dim1(npc2_);
#pragma omp simd simdlen(SIMD_WIDTH)
  for (int d = 0; d < ncloud_slice_; ++d) {
    ipc2v_slice_[d] = d % dim1;
    ipc3v_slice_[d] = d / dim1;
  }

#pragma omp simd simdlen(SIMD_WIDTH)
  for (int d = 0; d < ncloud_; ++d) {
    ipc1v_[d] = d / ncloud_slice_;
    islice_[d] = d % ncloud_slice_;
    ipc2v_[d] = ipc2v_slice_[islice_[d]];
    ipc3v_[d] = ipc3v_slice_[islice_[d]];
  }

  initialized_ = true;
}

void ParticleMesh::AddWeight() {// Add weight in meshaux.
  iweight = AddMeshAux();
  return;
}

//--------------------------------------------------------------------------------------
//! \fn int ParticleMesh::AddMeshAux()
//  \brief adds one auxiliary to the mesh and returns the index.

int ParticleMesh::AddMeshAux() { return nmeshaux++; }

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::InitIntermAux(const Real& value)
//  \brief updates all elements in the interm_aux_3d_ with value.

void ParticleMesh::InitIntermAux(const Real& value) {
  if (active3_) {
#pragma ivdep
    std::fill(
        &interm_aux_3d_[0][0][0],
        &interm_aux_3d_[0][0][0] + nx3_eff_ * nx2_eff_ * nx1_eff_ * NPC_SQR *
                                       InfoParticleMesh::nprop_aux_plus_one,
        value);
  } else if (active2_) {
#pragma ivdep
    std::fill(
        &interm_aux_2d_[0][0][0],
        &interm_aux_2d_[0][0][0] +
            nx2_eff_ * nx1_eff_ * NPC * InfoParticleMesh::nprop_aux_plus_one,
        value);
  } else {
#pragma ivdep
    std::fill(
        &interm_aux_1d_[0][0],
        &interm_aux_1d_[0][0] + nx1_eff_ * InfoParticleMesh::nprop_aux_plus_one,
        value);
  }
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::InitIntermBkgnd3D(const AthenaArray<Real> &meshsrc)
//  \brief transpose meshsrc to background_3d_.

void ParticleMesh::InitIntermBkgnd3D(const AthenaArray<Real>& meshsrc) {
  int imb2(0), imb3(0), ix(0);

  // Copies for vectorization
  InfoParticleMesh::IntermediateBackground3D* __restrict__ background_copy =
      background_3d_;
#pragma ivdep
  for (int k = 0; k < nx3_eff_; ++k) {
    int tmp1(k * nx2_eff_);  // reduce repeated calculation
    for (int j = 0; j < nx2_eff_; ++j) {
      int tmp2((tmp1 + j) * nx1_eff_);  // reduce repeated calculation
      for (int i = 0; i < nx1_eff_; ++i) {
        // Code the location in 1d.
        ix = tmp2 + i;
#pragma loop count(NPC_SQR)
        for (int d = 0; d < NPC_SQR; ++d) {
          imb2 = j + ipc2v_slice_[d];
          imb3 = k + ipc3v_slice_[d];
#pragma loop count(InfoParticleMesh::nprop_bg)
          for (int n = 0; n < InfoParticleMesh::nprop_bg; ++n)
            background_copy[ix][d][n] = meshsrc(n, imb3, imb2, i);
        }
      }
    }
  }
}

void ParticleMesh::InitIntermBkgnd2D(const AthenaArray<Real>& meshsrc) {
  int imb2(0), ix(0);
  const int imb3(pmb_->ks);
  // Copies for vectorization
  InfoParticleMesh::IntermediateBackground2D* __restrict__ background_copy =
      background_2d_;
#pragma ivdep
  for (int j = 0; j < nx2_eff_; ++j) {
    int tmp2(j * nx1_eff_);  // reduce repeated calculation
    for (int i = 0; i < nx1_eff_; ++i) {
      // Code the location in 1d.
      ix = tmp2 + i;
#pragma loop count(NPC)
      for (int d = 0; d < NPC; ++d) {
        imb2 = j + ipc2v_slice_[d];
#pragma loop count(InfoParticleMesh::nprop_bg)
        for (int n = 0; n < InfoParticleMesh::nprop_bg; ++n)
          background_copy[ix][d][n] = meshsrc(n, imb3, imb2, i);
      }
    }
  }
}

void ParticleMesh::InitIntermBkgnd1D(const AthenaArray<Real>& meshsrc) {
  const int imb2(pmb_->js), imb3(pmb_->ks);
  // Copies for vectorization
  InfoParticleMesh::IntermediateBackground1D* __restrict__ background_copy =
      background_1d_;
 
#pragma ivdep
  for (int i = 0; i < nx1_eff_; ++i) {
    // Code the location in 1d.
#pragma loop count(InfoParticleMesh::nprop_bg)
    for (int n = 0; n < InfoParticleMesh::nprop_bg; ++n)
      background_copy[i][n] = meshsrc(n, imb3, imb2, i);
  }

}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::GetWeightTSC(const Real &xi1, const Real &xi2,
//                             const Real &xi3)
//  \brief calculate w1_[][simd_loc], w_slice_[][simd_loc] and ixv_[simd_loc].

void ParticleMesh::GetWeightTSC(const Real& xi1, const Real& xi2,
                                const Real& xi3, const int& simd_loc) {
  // Find the domain the particle influences.
  int ix1(static_cast<int>(xi1 - dxi1_)), ix2(static_cast<int>(xi2 - dxi2_)),
      ix3(static_cast<int>(xi3 - dxi3_));

  if (active3_) {
    Real w2[NPC], w3[NPC];
    Real dxi3(xi3 - ix3 - 0.5 - dxi3_);
    w3[0] = 0.5 * SQR(0.5 - dxi3);
    w3[1] = 0.75 - SQR(dxi3);
    w3[2] = 0.5 * SQR(0.5 + dxi3);
    Real dxi2(xi2 - ix2 - 0.5 - dxi2_);
    w2[0] = 0.5 * SQR(0.5 - dxi2);
    w2[1] = 0.75 - SQR(dxi2);
    w2[2] = 0.5 * SQR(0.5 + dxi2);
    w_slice_[0][simd_loc] = w2[0] * w3[0];
    w_slice_[1][simd_loc] = w2[1] * w3[0];
    w_slice_[2][simd_loc] = w2[2] * w3[0];
    w_slice_[3][simd_loc] = w2[0] * w3[1];
    w_slice_[4][simd_loc] = w2[1] * w3[1];
    w_slice_[5][simd_loc] = w2[2] * w3[1];
    w_slice_[6][simd_loc] = w2[0] * w3[2];
    w_slice_[7][simd_loc] = w2[1] * w3[2];
    w_slice_[8][simd_loc] = w2[2] * w3[2];
  } else if (active2_) {
    Real dxi2(xi2 - ix2 - 0.5 - dxi2_);
    w_slice_[0][simd_loc] = 0.5 * SQR(0.5 - dxi2);
    w_slice_[1][simd_loc] = 0.75 - SQR(dxi2);
    w_slice_[2][simd_loc] = 0.5 * SQR(0.5 + dxi2);
  } else {
    w_slice_[0][simd_loc] = 1.0;
  }

  Real dxi1(xi1 - ix1 - 0.5 - dxi1_);
  w1_[0][simd_loc] = 0.5 * SQR(0.5 - dxi1);
  w1_[1][simd_loc] = 0.75 - SQR(dxi1);
  w1_[2][simd_loc] = 0.5 * SQR(0.5 + dxi1);
  // Code the location of the particle on the meshblock, in 1d.
  ixv_[simd_loc] = (ix3 * nx2_eff_ + ix2) * nx1_eff_ + ix1;
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::Interpolate4IntermBkgnd3D(const int& simd_loc,)
//  \brief Interpolate after GetWeightTSC.

void ParticleMesh::Interpolate4IntermBkgnd3D(const int& simd_loc) { 
  Real __attribute__((aligned(CACHELINE_BYTES)))
  bkgnd_slice[NPC][InfoParticleMesh::nprop_bg] = {{0.0}};  // Initialize
  InfoParticleMesh::IntermediateBackground3D* __restrict__ background_copy =
      background_3d_;
#pragma vector always aligned
#pragma loop count(NPC_SQR)
  for (int d = 0; d < NPC_SQR; ++d) {
#pragma loop count(InfoParticleMesh::nprop_bg)
    for (int n = 0; n < InfoParticleMesh::nprop_bg; ++n) {
      bkgnd_slice[0][n] +=
          w_slice_[d][simd_loc] * background_copy[ixv_[simd_loc]][d][n];
#pragma distribute point
      bkgnd_slice[1][n] +=
          w_slice_[d][simd_loc] * background_copy[ixv_[simd_loc] + 1][d][n];
#pragma distribute point
      bkgnd_slice[2][n] +=
          w_slice_[d][simd_loc] * background_copy[ixv_[simd_loc] + 2][d][n];
    }
  }


#pragma loop count(InfoParticleMesh::nprop_bg)
  for (int n = 0; n < InfoParticleMesh::nprop_bg; ++n)
    ppar_->bkgnd[n][simd_loc] = w1_[0][simd_loc] * bkgnd_slice[0][n] +
                                w1_[1][simd_loc] * bkgnd_slice[1][n] +
                                w1_[2][simd_loc] * bkgnd_slice[2][n];
 }

void ParticleMesh::Interpolate4IntermBkgnd2D(const int& simd_loc) {
   Real __attribute__((aligned(CACHELINE_BYTES)))
   bkgnd_slice[NPC][InfoParticleMesh::nprop_bg] = {{0.0}};  // Initialize
   InfoParticleMesh::IntermediateBackground2D* __restrict__ background_copy =
       background_2d_;
#pragma vector always aligned
#pragma loop count(NPC)
   for (int d = 0; d < NPC; ++d) {
#pragma loop count(InfoParticleMesh::nprop_bg)
     for (int n = 0; n < InfoParticleMesh::nprop_bg; ++n) {
       bkgnd_slice[0][n] +=
           w_slice_[d][simd_loc] * background_copy[ixv_[simd_loc]][d][n];
#pragma distribute point
       bkgnd_slice[1][n] +=
           w_slice_[d][simd_loc] * background_copy[ixv_[simd_loc] + 1][d][n];
#pragma distribute point
       bkgnd_slice[2][n] +=
           w_slice_[d][simd_loc] * background_copy[ixv_[simd_loc] + 2][d][n];
     }
   }

#pragma ivdep
#pragma loop count(InfoParticleMesh::nprop_bg)
   for (int n = 0; n < InfoParticleMesh::nprop_bg; ++n)
     ppar_->bkgnd[n][simd_loc] = w1_[0][simd_loc] * bkgnd_slice[0][n] +
                                 w1_[1][simd_loc] * bkgnd_slice[1][n] +
                                 w1_[2][simd_loc] * bkgnd_slice[2][n];
 }

void ParticleMesh::Interpolate4IntermBkgnd1D(const int& simd_loc) {
   InfoParticleMesh::IntermediateBackground1D* __restrict__ background_copy =
       background_1d_;
#pragma ivdep
#pragma loop count(InfoParticleMesh::nprop_bg)
   for (int n = 0; n < InfoParticleMesh::nprop_bg; ++n)
     ppar_->bkgnd[n][simd_loc] =
         w1_[0][simd_loc] * background_1d_[ixv_[simd_loc]][n] +
         w1_[1][simd_loc] * background_1d_[ixv_[simd_loc] + 1][n] +
         w1_[2][simd_loc] * background_1d_[ixv_[simd_loc] + 2][n];
 }

//--------------------------------------------------------------------------------------
 //! \fn void ParticleMesh::Assign2IntermAux3D(const int &simd_loc, const int& nprop)
 //  \brief assign ppar_->deposit_var[][simd_loc] to the intermediate arrays,
//  when w1_[][simd_loc], w_slice_[][simd_loc] and ixv_[simd_loc] have been
//  calculated before the calling.

void ParticleMesh::Assign2IntermAux3D(const int& simd_loc, const int& nprop) {
   InfoParticleMesh::IntermediateAuxiliary3D* __restrict__ interm_aux_copy =
       interm_aux_3d_;
   Real tmp;
   const int ix(ixv_[simd_loc]);
#pragma vector always aligned
#pragma loop count(NPC_SQR)
   for (int d = 0; d < NPC_SQR; ++d) {
#pragma loop count(InfoParticleMesh::nprop_aux)
     for (int n = 1; n < (nprop + 1); ++n) {
       tmp = w_slice_[d][simd_loc] * ppar_->deposit_var[n - 1][simd_loc];
       interm_aux_copy[ix][d][n] += tmp * w1_[0][simd_loc];
       interm_aux_copy[ix + 1][d][n] += tmp * w1_[1][simd_loc];
       interm_aux_copy[ix + 2][d][n] += tmp * w1_[2][simd_loc];
     }
   }
 }

void ParticleMesh::Assign2IntermAux2D(const int& simd_loc, const int& nprop) {
   InfoParticleMesh::IntermediateAuxiliary2D* __restrict__ interm_aux_copy =
       interm_aux_2d_;
   Real tmp;
   const int ix(ixv_[simd_loc]);
#pragma vector always aligned
#pragma loop count(NPC)
   for (int d = 0; d < NPC; ++d) {
#pragma loop count(InfoParticleMesh::nprop_aux)
     for (int n = 1; n < (nprop + 1); ++n) {
       tmp = w_slice_[d][simd_loc] * ppar_->deposit_var[n - 1][simd_loc];
       interm_aux_copy[ix][d][n] += tmp * w1_[0][simd_loc];
       interm_aux_copy[ix + 1][d][n] += tmp * w1_[1][simd_loc];
       interm_aux_copy[ix + 2][d][n] += tmp * w1_[2][simd_loc];
     }
   }
 }

void ParticleMesh::Assign2IntermAux1D(const int& simd_loc, const int& nprop) {
   InfoParticleMesh::IntermediateAuxiliary1D* __restrict__ interm_aux_copy =
       interm_aux_1d_;
   const int ix(ixv_[simd_loc]);
#pragma novector
#pragma loop count(NPC)
   for (int d = 0; d < NPC; ++d) {
#pragma loop count(InfoParticleMesh::nprop_aux)
     for (int n = 1; n < (nprop + 1); ++n) {
       interm_aux_copy[ix + d][n] +=
           w1_[d][simd_loc] * ppar_->deposit_var[n - 1][simd_loc];
     }
   }
 }

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::Assign2IntermAuxAndWght3D(const int &simd_loc, const int& nprop)
//  \brief assign weight additionally.

void ParticleMesh::Assign2IntermAuxAndWght3D(const int& simd_loc,
                                              const int& nprop) {
   InfoParticleMesh::IntermediateAuxiliary3D* __restrict__ interm_aux_copy =
       interm_aux_3d_;
   // Real tmp1, tmp2, tmp3;
   const int ix(ixv_[simd_loc]);
#pragma vector always aligned
#pragma loop count(NPC_SQR)
   for (int d = 0; d < NPC_SQR; ++d) {
     interm_aux_copy[ix][d][0] += w1_[0][simd_loc] * w_slice_[d][simd_loc];
#pragma loop count(InfoParticleMesh::nprop_aux)
     for (int n = 1; n < (nprop + 1); ++n) {
       interm_aux_copy[ix][d][n] += w1_[0][simd_loc] * w_slice_[d][simd_loc] *
                                    ppar_->deposit_var[n - 1][simd_loc];
     }
     interm_aux_copy[ix + 1][d][0] += w1_[1][simd_loc] * w_slice_[d][simd_loc];
#pragma loop count(InfoParticleMesh::nprop_aux)
     for (int n = 1; n < (nprop + 1); ++n) {
       interm_aux_copy[ix + 1][d][n] += w1_[1][simd_loc] *
                                        w_slice_[d][simd_loc] *
                                        ppar_->deposit_var[n - 1][simd_loc];
     }
     interm_aux_copy[ix + 2][d][0] += w1_[2][simd_loc] * w_slice_[d][simd_loc];
#pragma loop count(InfoParticleMesh::nprop_aux)
     for (int n = 1; n < (nprop + 1); ++n) {
       interm_aux_copy[ix + 2][d][n] += w1_[2][simd_loc] *
                                        w_slice_[d][simd_loc] *
                                        ppar_->deposit_var[n - 1][simd_loc];
     }
   }
 }

void ParticleMesh::Assign2IntermAuxAndWght2D(const int& simd_loc,
                                              const int& nprop) {
   InfoParticleMesh::IntermediateAuxiliary2D* __restrict__ interm_aux_copy =
       interm_aux_2d_;
   // Real tmp1, tmp2, tmp3;
   const int ix(ixv_[simd_loc]);
#pragma vector always aligned
#pragma loop count(NPC)
   for (int d = 0; d < NPC; ++d) {
     interm_aux_copy[ix][d][0] += w1_[0][simd_loc] * w_slice_[d][simd_loc];
#pragma loop count(InfoParticleMesh::nprop_aux, 3)
     for (int n = 1; n < (nprop + 1); ++n) {
       interm_aux_copy[ix][d][n] += w1_[0][simd_loc] * w_slice_[d][simd_loc] *
                                    ppar_->deposit_var[n - 1][simd_loc];
     }
     interm_aux_copy[ix + 1][d][0] += w1_[1][simd_loc] * w_slice_[d][simd_loc];
#pragma loop count(InfoParticleMesh::nprop_aux, 3)
     for (int n = 1; n < (nprop + 1); ++n) {
       interm_aux_copy[ix + 1][d][n] += w1_[1][simd_loc] *
                                        w_slice_[d][simd_loc] *
                                        ppar_->deposit_var[n - 1][simd_loc];
     }
     interm_aux_copy[ix + 2][d][0] += w1_[2][simd_loc] * w_slice_[d][simd_loc];
#pragma loop count(InfoParticleMesh::nprop_aux, 3)
     for (int n = 1; n < (nprop + 1); ++n) {
       interm_aux_copy[ix + 2][d][n] += w1_[2][simd_loc] *
                                        w_slice_[d][simd_loc] *
                                        ppar_->deposit_var[n - 1][simd_loc];
     }
   }
 }

void ParticleMesh::Assign2IntermAuxAndWght1D(const int& simd_loc,
                                              const int& nprop) {
   InfoParticleMesh::IntermediateAuxiliary1D* __restrict__ interm_aux_copy =
       interm_aux_1d_;
   // Real tmp1, tmp2, tmp3;
   const int ix(ixv_[simd_loc]);
#pragma novector
#pragma loop count(NPC)
   for (int d = 0; d < NPC; ++d) {
     interm_aux_copy[ix + d][0] += w1_[d][simd_loc];
#pragma loop count(InfoParticleMesh::nprop_aux, 3)
     for (int n = 1; n < (nprop + 1); ++n) {
       interm_aux_copy[ix + d][n] +=
           w1_[d][simd_loc] * ppar_->deposit_var[n - 1][simd_loc];
     }
   }
 }

//--------------------------------------------------------------------------------------
//! \fn ParticleMesh::ParticleMesh(Particles *ppar, int nmeshaux)
//  \brief constructs a new ParticleMesh instance.

ParticleMesh::ParticleMesh(Particles* ppar)
    : ppar_(ppar),
      pmb_(ppar->pmy_block),
      pmesh_(ppar->pmy_block->pmy_mesh),
      nx1_(ppar->pmy_block->ncells1),
      nx2_(ppar->pmy_block->ncells2),
      nx3_(ppar->pmy_block->ncells3),
      ncells_(ppar->pmy_block->ncells1 * ppar->pmy_block->ncells2 *
              ppar->pmy_block->ncells3),
      meshaux(nmeshaux, ppar->pmy_block->ncells3, ppar->pmy_block->ncells2,
              ppar->pmy_block->ncells1,
              (nmeshaux > 0) ? AthenaArray<Real>::DataStatus::allocated
                             : AthenaArray<Real>::DataStatus::empty),
      vol(ppar->pmy_block->ncells1),
      is(ppar->pmy_block->is),
      ie(ppar->pmy_block->ie),
      js(ppar->pmy_block->js),
      je(ppar->pmy_block->je),
      ks(ppar->pmy_block->ks),
      ke(ppar->pmy_block->ke),
      nx3_eff_(nx3_ - npc3_ + 1),
      nx2_eff_(nx2_ - npc2_ + 1),
      nx1_eff_(nx1_),
      pmbvar_(ppar->pmy_block, &meshaux) {

  // Get a shorthand to weights.
  if (ParticleMesh::iweight >= 0)
    weight.InitWithShallowSlice(meshaux, 4, iweight, 1);

  // Enroll ParMeshBoundaryVariable object
  if (ppar->backreaction) {
    pmbvar_.bvar_index = pmb_->pbval->bvars.size();
    pmb_->pbval->bvars.push_back(&pmbvar_);
    pmb_->pbval->bvars_main_int.push_back(&pmbvar_);
  }
  // Even though "ApplyPhysicalBoundaries" means different for particle-mesh and
  // hydro, where particle-mesh needs to apply physical boundaries before update
  // particle source term. But it is ok to call "ApplyPhysicalBoundaries" twice
  // for particle-mesh.

  // Intiate the intermediate arrays and ipc
  int err(0);  // check whether posix_memalign succeeds
  if (active3_) {
    const InfoParticleMesh::IntermediateBackground3D intermediate_background =
        {};
    err += posix_memalign(
        reinterpret_cast<void**>(&background_3d_), CACHELINE_BYTES,
        sizeof(intermediate_background) * nx3_eff_ * nx2_eff_ * nx1_eff_);
  } else if (active2_) {
    const InfoParticleMesh::IntermediateBackground2D intermediate_background =
        {};
    err += posix_memalign(
        reinterpret_cast<void**>(&background_2d_), CACHELINE_BYTES,
        sizeof(intermediate_background) * nx2_eff_ * nx1_eff_);
  } else {
    const InfoParticleMesh::IntermediateBackground1D intermediate_background =
        {};
    err += posix_memalign(reinterpret_cast<void**>(&background_1d_),
                          CACHELINE_BYTES,
                          sizeof(intermediate_background) * nx1_eff_);
  }
  if (ppar->backreaction) {
    if (active3_) {
      const InfoParticleMesh::IntermediateAuxiliary3D intermediate_auxiliary =
          {};
      err += posix_memalign(
          reinterpret_cast<void**>(&interm_aux_3d_), CACHELINE_BYTES,
          sizeof(intermediate_auxiliary) * nx3_eff_ * nx2_eff_ * nx1_eff_);
    } else if (active2_) {
      const InfoParticleMesh::IntermediateAuxiliary2D intermediate_auxiliary =
          {};
      err += posix_memalign(
          reinterpret_cast<void**>(&interm_aux_2d_), CACHELINE_BYTES,
          sizeof(intermediate_auxiliary) * nx2_eff_ * nx1_eff_);
    } else {
      const InfoParticleMesh::IntermediateAuxiliary1D intermediate_auxiliary =
          {};
      err += posix_memalign(reinterpret_cast<void**>(&interm_aux_1d_),
                            CACHELINE_BYTES,
                            sizeof(intermediate_auxiliary) * nx1_eff_);
    }
  }
  if (err != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ParticleMesh::ParticleMesh" << std::endl
        << "Fail to allocate space." << std::endl;
    ATHENA_ERROR(msg);
  }
}

//--------------------------------------------------------------------------------------
//! \fn ParticleMesh::~ParticleMesh()
//  \brief destructs a ParticleMesh instance.

ParticleMesh::~ParticleMesh() {
  // Destroy the particle meshblock.
  weight.DeleteAthenaArray();
  meshaux.DeleteAthenaArray();
  vol.DeleteAthenaArray();

  // Release the intermediate arrays
  if (active3_) {
    free(background_3d_);
  } else if (active2_) {
    free(background_2d_);
  } else {
    free(background_1d_);
  }
  if (ppar_->backreaction) {
    if (active3_) {
      free(interm_aux_3d_);
    } else if (active2_) {
      free(interm_aux_2d_);
    } else {
      free(interm_aux_1d_);
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn Real ParticleMesh::FindMaximumWeight()
//  \brief returns the maximum weight in the meshblock.

Real ParticleMesh::FindMaximumWeight() const {
  if (iweight < 0) return 0;
  Real wmax = 0.0;
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i) wmax = std::max(wmax, weight(k, j, i));
  return wmax;
}

//--------------------------------------------------------------------------------------
//! \fn int ParticleMesh::GetCellNumDim1() const
//  \brief return nx1_.

int ParticleMesh::GetCellNumDim1() const { return nx1_; }

//--------------------------------------------------------------------------------------
//! \fn int ParticleMesh::GetCellNumDim2() const
//  \brief return nx2_.

int ParticleMesh::GetCellNumDim2() const { return nx2_; }

//--------------------------------------------------------------------------------------
//! \fn int ParticleMesh::GetCellNumDim3() const
//  \brief return nx3_.

int ParticleMesh::GetCellNumDim3() const { return nx3_; }

//--------------------------------------------------------------------------------------
//! \fn int ParticleMesh::GetTotalCellNum() const
//  \brief returns ncells_.

int ParticleMesh::GetTotalCellNum() const { return ncells_; }

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::InterpolateMeshToParticles(
//               const AthenaArray<Real>& meshsrc, int ms1,
//               AthenaArray<Real>& par, int p1, int nprop)
//  \brief interpolates meshsrc from property index ms1 to ms1+nprop-1 onto
//  particle
//      array par (realprop, auxprop, or work in Particles class) from property
//      index p1 to p1+nprop-1.

void ParticleMesh::InterpolateMeshToParticles(const AthenaArray<Real>& meshsrc,
                                              int ms1, AthenaArray<Real>& par,
                                              int p1, int nprop) {
  // Zero out the particle arrays.
  for (int n = 0; n < nprop; ++n)
#pragma ivdep
    std::fill(&par(p1 + n, 0), &par(p1 + n, ppar_->npar), 0.0);

  // Loop over each particle.
  for (int k = 0; k < ppar_->npar; ++k) {
    // Find the domain the particle influences.
    Real xi1 = ppar_->xi1(k), xi2 = ppar_->xi2(k), xi3 = ppar_->xi3(k);
    int ix1 = static_cast<int>(xi1 - dxi1_),
        ix2 = static_cast<int>(xi2 - dxi2_),
        ix3 = static_cast<int>(xi3 - dxi3_);
    xi1 = ix1 + 0.5 - xi1;
    xi2 = ix2 + 0.5 - xi2;
    xi3 = ix3 + 0.5 - xi3;

    // Weight each cell and accumulate the mesh properties onto the particles.
#pragma loop count(NPC)
    for (int ipc3 = 0; ipc3 < npc3_; ++ipc3) {
#pragma loop count(NPC)
      for (int ipc2 = 0; ipc2 < npc2_; ++ipc2) {
#pragma loop count(NPC)
        for (int ipc1 = 0; ipc1 < npc1_; ++ipc1) {
          Real w = (active1_ ? _WeightFunction(xi1 + ipc1) : 1.0) *
                   (active2_ ? _WeightFunction(xi2 + ipc2) : 1.0) *
                   (active3_ ? _WeightFunction(xi3 + ipc3) : 1.0);
          int imb1 = ix1 + ipc1, imb2 = ix2 + ipc2, imb3 = ix3 + ipc3;

          for (int n = 0; n < nprop; ++n)
            par(p1 + n, k) += w * meshsrc(ms1 + n, imb3, imb2, imb1);
        }
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::AssignParticlesToMeshAux(
//               const AthenaArray<Real>& par, int p1, int p2, int ma1)
//  \brief assigns par (realprop, auxprop, or work in Particles class) from
//  property
//         index p1 to p2 onto meshaux from property index ma1 and up.

void ParticleMesh::AssignParticlesToMeshAux(const AthenaArray<Real>& par,
                                            int p1, int ma1, int nprop) {
  // Prepare to transpose meshsrc and weight.
  AthenaArray<Real> aux(nx3_, nx2_, nx1_, nprop + 1);

  // Allocate space for SIMD.
  Real** w1 __attribute__((aligned(64))) = new Real*[npc1_];
  Real** w2 __attribute__((aligned(64))) = new Real*[npc2_];
  Real** w3 __attribute__((aligned(64))) = new Real*[npc3_];
  for (int i = 0; i < npc1_; ++i) w1[i] = new Real[SIMD_WIDTH];
  for (int i = 0; i < npc2_; ++i) w2[i] = new Real[SIMD_WIDTH];
  for (int i = 0; i < npc3_; ++i) w3[i] = new Real[SIMD_WIDTH];
  int ix1v[SIMD_WIDTH] __attribute__((aligned(64)));
  int ix2v[SIMD_WIDTH] __attribute__((aligned(64)));
  int ix3v[SIMD_WIDTH] __attribute__((aligned(64)));

  // Loop over each particle.
  int npar(ppar_->npar);
  for (int k = 0; k < npar; k += SIMD_WIDTH) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int kk = 0; kk < std::min(SIMD_WIDTH, npar - k); ++kk) {
      int kkk = k + kk;
      // Find the domain the particle influences.
      Real xi1(ppar_->xi1(kkk)), xi2(ppar_->xi2(kkk)), xi3(ppar_->xi3(kkk));
      int ix1(static_cast<int>(xi1 - dxi1_)),
          ix2(static_cast<int>(xi2 - dxi2_)),
          ix3(static_cast<int>(xi3 - dxi3_));
      xi1 = ix1 + 0.5 - xi1;
      xi2 = ix2 + 0.5 - xi2;
      xi3 = ix3 + 0.5 - xi3;
      ix1v[kk] = ix1;
      ix2v[kk] = ix2;
      ix3v[kk] = ix3;

      // Weigh each cell.
#pragma loop count(NPC)
      for (int i = 0; i < npc1_; ++i)
        w1[i][kk] = active1_ ? _WeightFunction(xi1 + i) : 1.0;
#pragma loop count(NPC)
      for (int i = 0; i < npc2_; ++i)
        w2[i][kk] = active2_ ? _WeightFunction(xi2 + i) : 1.0;
#pragma loop count(NPC)
      for (int i = 0; i < npc3_; ++i)
        w3[i][kk] = active3_ ? _WeightFunction(xi3 + i) : 1.0;
    }

    for (int kk = 0; kk < std::min(SIMD_WIDTH, npar - k); ++kk) {
      int kkk = k + kk;

      // Fetch properties of the particle for assignment.
      Real* p = new Real[nprop];
      for (int n = 0; n < nprop; ++n) p[n] = par(p1 + n, kkk);

      // Weight each cell and accumulate particle property onto meshaux.
      int ix1(ix1v[kk]), ix2(ix2v[kk]), ix3(ix3v[kk]);

#pragma loop count(NPC)
      for (int ipc3 = 0; ipc3 < npc3_; ++ipc3) {
#pragma loop count(NPC)
        for (int ipc2 = 0; ipc2 < npc2_; ++ipc2) {
#pragma loop count(NPC)
          for (int ipc1 = 0; ipc1 < npc1_; ++ipc1) {
            Real w(w1[ipc1][kk] * w2[ipc2][kk] * w3[ipc3][kk]);
            int ima1(ix1 + ipc1), ima2(ix2 + ipc2), ima3(ix3 + ipc3);

            for (int n = 0; n < nprop; ++n)
              aux(ima3, ima2, ima1, n) += w * p[n];
            aux(ima3, ima2, ima1, nprop) += w;
          }
        }
      }
      delete[] p;
    }
  }

  // Transpose back to weight and meshaux.
  for (size_t k = 0; k < nx3_; ++k) {
    for (size_t j = 0; j < nx2_; ++j) {
      for (size_t i = 0; i < nx1_; ++i) {
        for (size_t n = 0; n < nprop; ++n) {
          meshaux(ma1 + n, k, j, i) = aux(k, j, i, n);
        }
        weight(k, j, i) = aux(k, j, i, nprop);
      }
    }
  }

  // Release working array.
  for (int i = 0; i < npc1_; ++i) delete[] w1[i];
  for (int i = 0; i < npc2_; ++i) delete[] w2[i];
  for (int i = 0; i < npc3_; ++i) delete[] w3[i];
  delete[] w1;
  delete[] w2;
  delete[] w3;
  // Treat neighbors of different levels.
  if (pmesh_->multilevel) AssignParticlesToDifferentLevels(par, p1, ma1, nprop);
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::InterpolateMeshAndAssignParticles(
//               const AthenaArray<Real>& meshsrc, int ms1,
//               AthenaArray<Real>& pardst, int pd1, int ni,
//               const AthenaArray<Real>& parsrc, int ps1, int ma1, int na)
//  \brief interpolates meshsrc from property index ms1 to ms1 + ni - 1 onto
//  particle
//      array pardst from index pd1 to pd1 + ni - 1, and assigns parsrc from
//      property index ps1 to ps1 + na - 1 onto meshaux from ma1 to ma1 + na
//      - 1.  The arrays parsrc and pardst can be realprop, auxprop, or work in
//      Particles class.

void ParticleMesh::InterpolateMeshAndAssignParticles(
    const AthenaArray<Real>& meshsrc, int ms1, AthenaArray<Real>& pardst,
    int pd1, int ni, const AthenaArray<Real>& parsrc, int ps1, int ma1,
    int na) { 
  // Zero out meshaux.
#pragma ivdep
  meshaux.ZeroClear();

  // Transpose meshsrc.
  int nx1 = meshsrc.GetDim1(), nx2 = meshsrc.GetDim2(), nx3 = meshsrc.GetDim3();
  AthenaArray<Real> u;
  u.NewAthenaArray(nx3, nx2, nx1, ni);
  for (int n = 0; n < ni; ++n)
    for (int k = 0; k < nx3; ++k)
      for (int j = 0; j < nx2; ++j)
        for (int i = 0; i < nx1; ++i) u(k, j, i, n) = meshsrc(ms1 + n, k, j, i);

  // Allocate space for SIMD.
  Real** w1 __attribute__((aligned(64))) = new Real*[npc1_];
  Real** w2 __attribute__((aligned(64))) = new Real*[npc2_];
  Real** w3 __attribute__((aligned(64))) = new Real*[npc3_];
  for (int i = 0; i < npc1_; ++i) w1[i] = new Real[SIMD_WIDTH];
  for (int i = 0; i < npc2_; ++i) w2[i] = new Real[SIMD_WIDTH];
  for (int i = 0; i < npc3_; ++i) w3[i] = new Real[SIMD_WIDTH];
  Real imb1v[SIMD_WIDTH] __attribute__((aligned(64)));
  Real imb2v[SIMD_WIDTH] __attribute__((aligned(64)));
  Real imb3v[SIMD_WIDTH] __attribute__((aligned(64)));

  // Loop over each particle.
  int npar = ppar_->npar;
  for (int k = 0; k < npar; k += SIMD_WIDTH) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int kk = 0; kk < std::min(SIMD_WIDTH, npar - k); ++kk) {
      int kkk = k + kk;

      // Find the domain the particle influences.
      Real xi1 = ppar_->xi1(kkk), xi2 = ppar_->xi2(kkk), xi3 = ppar_->xi3(kkk);
      int imb1 = static_cast<int>(xi1 - dxi1_),
          imb2 = static_cast<int>(xi2 - dxi2_),
          imb3 = static_cast<int>(xi3 - dxi3_);
      xi1 = imb1 + 0.5 - xi1;
      xi2 = imb2 + 0.5 - xi2;
      xi3 = imb3 + 0.5 - xi3;

      imb1v[kk] = imb1;
      imb2v[kk] = imb2;
      imb3v[kk] = imb3;

      // Weigh each cell.
#pragma loop count(NPC)
      for (int i = 0; i < npc1_; ++i)
        w1[i][kk] = active1_ ? _WeightFunction(xi1 + i) : 1.0;
#pragma loop count(NPC)
      for (int i = 0; i < npc2_; ++i)
        w2[i][kk] = active2_ ? _WeightFunction(xi2 + i) : 1.0;
#pragma loop count(NPC)
      for (int i = 0; i < npc3_; ++i)
        w3[i][kk] = active3_ ? _WeightFunction(xi3 + i) : 1.0;
    }

#pragma ivdep
    for (int kk = 0; kk < std::min(SIMD_WIDTH, npar - k); ++kk) {
      int kkk = k + kk;

      // Initiate interpolation and fetch particle properties.
      Real* pd = new Real[ni];
      Real* ps = new Real[na];
      for (int i = 0; i < ni; ++i) pd[i] = 0.0;
      for (int i = 0; i < na; ++i) ps[i] = parsrc(ps1 + i, kkk);

      int imb1 = imb1v[kk], imb2 = imb2v[kk], imb3 = imb3v[kk];

#pragma loop count(NPC)
      for (int ipc3 = 0; ipc3 < npc3_; ++ipc3) {
#pragma loop count(NPC)
        for (int ipc2 = 0; ipc2 < npc2_; ++ipc2) {
#pragma loop count(NPC)
          for (int ipc1 = 0; ipc1 < npc1_; ++ipc1) {
            Real w(w1[ipc1][kk] * w2[ipc2][kk] * w3[ipc3][kk]);

            // Record the weights.
            weight(imb3 + ipc3, imb2 + ipc2, imb1 + ipc1) += w;

            // Interpolate meshsrc to particles.
            for (int n = 0; n < ni; ++n)
              pd[n] += w * u(imb3 + ipc3, imb2 + ipc2, imb1 + ipc1, n);

            // Assign particles to meshaux.
            for (int n = 0; n < na; ++n)
              meshaux(ma1 + n, imb3 + ipc3, imb2 + ipc2, imb1 + ipc1) +=
                  w * ps[n];
          }
        }
      }

      // Record the final interpolated properties.
      for (int n = 0; n < ni; ++n) pardst(pd1 + n, kkk) = pd[n];

      delete[] pd;
      delete[] ps;
    }
  }

  // Release working array.
  u.DeleteAthenaArray();
  for (int i = 0; i < npc1_; ++i) delete[] w1[i];
  for (int i = 0; i < npc2_; ++i) delete[] w2[i];
  for (int i = 0; i < npc3_; ++i) delete[] w3[i];
  delete[] w1;
  delete[] w2;
  delete[] w3;

  // Treat neighbors of different levels.
  if (pmesh_->multilevel)
    AssignParticlesToDifferentLevels(parsrc, ps1, ma1, na);
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::AssignIntermAux3D2MeshAux((const int& ma,
//                                             const int& begin,
//                                             const int& end,
//                                             const int& last,const bool&
//                                             flag_rho)
//  \brief deposit ppar_->interm_aux_3d_ to meshaux and weight

void ParticleMesh::AssignIntermAux3D2MeshAux(const int& ma, const int& begin,
                                           const int& nprop, const int& last,
                                             const bool& flag_rho) {
#pragma ivdep
  meshaux.ZeroClear();

  int imb2(0), imb3(0), ix(0);
  // Shallow copy for vectorization (cheating the compiler)
  AthenaArray<Real> shallow_weight, shallow_meshaux;
  shallow_weight.InitWithShallowSlice(meshaux, 4, iweight, 1);
  shallow_meshaux.InitWithShallowSlice(meshaux, 4, ma, nprop);

  InfoParticleMesh::IntermediateAuxiliary3D* __restrict__ interm_aux_copy =
      interm_aux_3d_;

 #pragma ivdep
  for (int k = 0; k < nx3_eff_; ++k) {
    int tmp1(k * nx2_eff_);  // reduce repeated calculation
    for (int j = 0; j < nx2_eff_; ++j) {
      int tmp2((tmp1 + j) * nx1_eff_);  // reduce repeated calculation
      for (int i = 0; i < nx1_eff_; ++i) {
        ix = tmp2 + i;
        // Accumulate the interm_aux_3d_
#pragma loop count(NPC_SQR)
        for (int d = 0; d < NPC_SQR; ++d) {
          imb2 = j + ipc2v_slice_[d];
          imb3 = k + ipc3v_slice_[d];

#pragma loop count(InfoParticleMesh::nprop_aux)
          for (int n = 0; n < nprop; ++n)
            shallow_meshaux(n, imb3, imb2, i) +=
                interm_aux_copy[ix][d][n + begin + 1];
          if (flag_rho)
            shallow_weight(imb3, imb2, i) += interm_aux_copy[ix][d][0];
        }
      }
    }
  }

  shallow_weight.DeleteAthenaArray();
  shallow_meshaux.DeleteAthenaArray();
}

void ParticleMesh::AssignIntermAux2D2MeshAux(const int& ma, const int& begin,
                                             const int& nprop, const int& last,
                                             const bool& flag_rho) {
#pragma ivdep
  meshaux.ZeroClear();

  int imb2(0), ix(0);
  const int imb3(pmb_->ks);
  // Shallow copy for vectorization (cheating the compiler)
  AthenaArray<Real> shallow_weight, shallow_meshaux;
  shallow_weight.InitWithShallowSlice(meshaux, 4, iweight, 1);
  shallow_meshaux.InitWithShallowSlice(meshaux, 4, ma, nprop);

  InfoParticleMesh::IntermediateAuxiliary2D* __restrict__ interm_aux_copy =
      interm_aux_2d_;

#pragma ivdep
  for (int j = 0; j < nx2_eff_; ++j) {
    int tmp2(j * nx1_eff_);  // reduce repeated calculation
    for (int i = 0; i < nx1_eff_; ++i) {
      ix = tmp2 + i;
      // Accumulate the interm_aux_3d_
#pragma loop count(NPC)
      for (int d = 0; d < NPC; ++d) {
        imb2 = j + ipc2v_slice_[d];

#pragma loop count(InfoParticleMesh::nprop_aux)
        for (int n = 0; n < nprop; ++n)
          shallow_meshaux(n, imb3, imb2, i) +=
              interm_aux_copy[ix][d][n + begin + 1];
        if (flag_rho)
          shallow_weight(imb3, imb2, i) += interm_aux_copy[ix][d][0];
      }
    }
  }

  shallow_weight.DeleteAthenaArray();
  shallow_meshaux.DeleteAthenaArray();
}

void ParticleMesh::AssignIntermAux1D2MeshAux(const int& ma, const int& begin,
                                             const int& nprop, const int& last,
                                             const bool& flag_rho) {
#pragma ivdep
  meshaux.ZeroClear();

  const int imb2(pmb_->js), imb3(pmb_->ks);
  // Shallow copy for vectorization (cheating the compiler)
  AthenaArray<Real> shallow_weight, shallow_meshaux;
  shallow_weight.InitWithShallowSlice(meshaux, 4, iweight, 1);
  shallow_meshaux.InitWithShallowSlice(meshaux, 4, ma, nprop);

  InfoParticleMesh::IntermediateAuxiliary1D* __restrict__ interm_aux_copy =
      interm_aux_1d_;
#pragma ivdep
  for (int i = 0; i < nx1_eff_; ++i) {
#pragma loop count(InfoParticleMesh::nprop_aux)
    for (int n = 0; n < nprop; ++n)
      shallow_meshaux(n, imb3, imb2, i) += interm_aux_copy[i][n + begin + 1];
    if (flag_rho) shallow_weight(imb3, imb2, i) += interm_aux_copy[i][0];
  }

  shallow_weight.DeleteAthenaArray();
  shallow_meshaux.DeleteAthenaArray();
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::DepositMeshAux(AthenaArray<Real>& u,
//                                        int ma1, int mb1, int nprop)
//  \brief deposits data in meshaux from property index ma1 to ma1+nprop-1 to
//  meshblock data u from property index mb1 and mb1+nprop-1, divided by cell volume.

void ParticleMesh::DepositMeshAux(AthenaArray<Real>& u, int ma1, int mb1,
                                  int nprop) {
  for (int n = 0; n < nprop; ++n)
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j) {
        pmb_->pcoord->CellVolume(k, j, is, ie, vol);
#pragma omp simd simdlen(SIMD_WIDTH)
        for (int i = is; i <= ie; ++i)
          u(mb1 + n, k, j, i) += meshaux(ma1 + n, k, j, i) / vol(i);
      }
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::SetBoundaryAttributes()
//  \brief initializes or reinitializes attributes for each boundary.

void ParticleMesh::SetBoundaryAttributes() {
  const RegionSize& block_size = pmb_->block_size;
  const Real xi1mid = (is + ie + 1) / 2, xi2mid = (js + je + 1) / 2,
             xi3mid = (ks + ke + 1) / 2;
  const int mylevel = pmb_->loc.level;
  const int myfx1 = static_cast<int>(pmb_->pbval->loc.lx1 & 1L),
            myfx2 = static_cast<int>(pmb_->pbval->loc.lx2 & 1L),
            myfx3 = static_cast<int>(pmb_->pbval->loc.lx3 & 1L);
  const int NGH = (NGHOST + 1) / 2, NG2 = 2 * NGHOST;
  const int nx1h = active1_ ? block_size.nx1 / 2 + NGH : 1,
            nx2h = active2_ ? block_size.nx2 / 2 + NGH : 1,
            nx3h = active3_ ? block_size.nx3 / 2 + NGH : 1;

  // Loop over each neighbor block.
  for (int n = 0; n < pmb_->pbval->nneighbor; ++n) {
    NeighborBlock& nb = pmb_->pbval->neighbor[n];

    // Find the index domain.
    Real xi1min = is - 1, xi1max = ie + 2, xi2min = js - 1, xi2max = je + 2,
         xi3min = ks - 1, xi3max = ke + 2;
    Real xi1_0 = is, xi2_0 = js, xi3_0 = ks;

    // Find several depths from the neighbor block.
    Real dxip, dxig;
    int nblevel = nb.snb.level;
    if (nblevel > mylevel) {
      dxip = 0.5 * RINF;
      dxig = NGHOST;
    } else if (nblevel < mylevel) {
      dxip = 2 * RINF;
      dxig = 2 * NGH;
    } else {
      continue;
      /*dxip = RINF;
      dxig = NGHOST;*/
    }

    // Consider the normal directions.
    NeighborIndexes& ni = nb.ni;
    if (ni.ox1 > 0) {
      xi1min = ie + 1 - dxip;
      xi1_0 = ie + 1;
    } else if (ni.ox1 < 0) {
      xi1max = is + dxip;
      xi1_0 = is - dxig;
    }

    if (ni.ox2 > 0) {
      xi2min = je + 1 - dxip;
      xi2_0 = je + 1;
    } else if (ni.ox2 < 0) {
      xi2max = js + dxip;
      xi2_0 = js - dxig;
    }

    if (ni.ox3 > 0) {
      xi3min = ke + 1 - dxip;
      xi3_0 = ke + 1;
    } else if (ni.ox3 < 0) {
      xi3max = ks + dxip;
      xi3_0 = ks - dxig;
    }

    // Consider the transverse directions.
    if (nblevel > mylevel) {  // Neighbor block is at a finer level.
      if (ni.type == NeighborConnect::face) {
        if (ni.ox1 != 0) {
          if (active2_) {
            if (ni.fi1) {
              xi2min = xi2mid - dxip;
              xi2_0 = xi2mid;
            } else {
              xi2max = xi2mid + dxip;
            }
          }
          if (active3_) {
            if (ni.fi2) {
              xi3min = xi3mid - dxip;
              xi3_0 = xi3mid;
            } else {
              xi3max = xi3mid + dxip;
            }
          }
        } else if (ni.ox2 != 0) {
          if (active1_) {
            if (ni.fi1) {
              xi1min = xi1mid - dxip;
              xi1_0 = xi1mid;
            } else {
              xi1max = xi1mid + dxip;
            }
          }
          if (active3_) {
            if (ni.fi2) {
              xi3min = xi3mid - dxip;
              xi3_0 = xi3mid;
            } else {
              xi3max = xi3mid + dxip;
            }
          }
        } else {
          if (active1_) {
            if (ni.fi1) {
              xi1min = xi1mid - dxip;
              xi1_0 = xi1mid;
            } else {
              xi1max = xi1mid + dxip;
            }
          }
          if (active2_) {
            if (ni.fi2) {
              xi2min = xi2mid - dxip;
              xi2_0 = xi2mid;
            } else {
              xi2max = xi2mid + dxip;
            }
          }
        }
      } else if (ni.type == NeighborConnect::edge) {
        if (ni.ox1 == 0) {
          if (active1_) {
            if (ni.fi1) {
              xi1min = xi1mid - dxip;
              xi1_0 = xi1mid;
            } else {
              xi1max = xi1mid + dxip;
            }
          }
        } else if (ni.ox2 == 0) {
          if (active2_) {
            if (ni.fi1) {
              xi2min = xi2mid - dxip;
              xi2_0 = xi2mid;
            } else {
              xi2max = xi2mid + dxip;
            }
          }
        } else {
          if (active3_) {
            if (ni.fi1) {
              xi3min = xi3mid - dxip;
              xi3_0 = xi3mid;
            } else {
              xi3max = xi3mid + dxip;
            }
          }
        }
      }
    } else if (nblevel < mylevel) {  // Neighbor block is at a coarser level.
      if (ni.type == NeighborConnect::face) {
        if (ni.ox1 != 0) {
          if (active2_ && myfx2) xi2_0 = js - dxig;
          if (active3_ && myfx3) xi3_0 = ks - dxig;
        } else if (ni.ox2 != 0) {
          if (active1_ && myfx1) xi1_0 = is - dxig;
          if (active3_ && myfx3) xi3_0 = ks - dxig;
        } else {
          if (active1_ && myfx1) xi1_0 = is - dxig;
          if (active2_ && myfx2) xi2_0 = js - dxig;
        }
      } else if (ni.type == NeighborConnect::edge) {
        if (ni.ox1 == 0) {
          if (active1_ && myfx1) xi1_0 = is - dxig;
        } else if (ni.ox2 == 0) {
          if (active2_ && myfx2) xi2_0 = js - dxig;
        } else {
          if (active3_ && myfx3) xi3_0 = ks - dxig;
        }
      }
    }

    // Set the domain that influences the ghost block.
    BoundaryAttributes& ba = ba_[nb.bufid];
    ba.xi1min = xi1min;
    ba.xi1max = xi1max;
    ba.xi2min = xi2min;
    ba.xi2max = xi2max;
    ba.xi3min = xi3min;
    ba.xi3max = xi3max;

    // Set the origin of the ghost block.
    ba.xi1_0 = xi1_0;
    ba.xi2_0 = xi2_0;
    ba.xi3_0 = xi3_0;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::AssignParticlesToDifferentLevels( const
//! AthenaArray<Real>& par, int p1, int ma1, int nprop, const AthenaArray<int>&
//! tid, const std::vector<Real>& property, const AthenaArray<Real>
//! &weight_delta_f)
//  \brief assigns particle array par from property index p1 to p1+nprop-1 to
//  meshaux from property index ma1 to ma1+nprop-1 in neighbors of different
//  levels.

void ParticleMesh::AssignParticlesToDifferentLevels(
    const AthenaArray<Real>& par, int p1, int ma1, int nprop,
    const AthenaArray<int>& tid, const std::vector<Real>& property,
    const AthenaArray<Real>& weight_delta_f) {
  if (!ppar_->backreaction) return;
  pmbvar_.InitDiffLevelVar();

  const int mylevel = pmb_->loc.level;
  const bool mul_factor(!tid.IsEmpty() && (property.size() != 0));
  const bool delta_f_enable(!weight_delta_f.IsEmpty());
  AthenaArray<Real> dest_weight;

  for (int k = 0; k < ppar_->npar; ++k) {
    // Find particles that influences the neighbor block.
    for (int i = 0; i < pmb_->pbval->nneighbor; ++i) {
      Real xi1 = ppar_->xi1(k), xi2 = ppar_->xi2(k), xi3 = ppar_->xi3(k);
      NeighborBlock& nb = pmb_->pbval->neighbor[i];
      SimpleNeighborBlock& snb = nb.snb;
      if (snb.level == mylevel) continue;

      AthenaArray<Real>& destination = pmbvar_.diff_level_var[nb.bufid];
      dest_weight.InitWithShallowSlice(destination, 4, iweight, 1);
      BoundaryAttributes& ba = ba_[nb.bufid];

      if (active3_) {
        if (!(xi1 >= ba.xi1min && xi1 <= ba.xi1max && xi2 >= ba.xi2min &&
              xi2 <= ba.xi2max && xi3 >= ba.xi3min && xi3 <= ba.xi3max))
          continue;
      } else if (active2_) {
        if (!(xi1 >= ba.xi1min && xi1 <= ba.xi1max && xi2 >= ba.xi2min &&
              xi2 <= ba.xi2max))
          continue;
      } else {
        if (!(xi1 >= ba.xi1min && xi1 <= ba.xi1max)) continue;
      }

      // Shift and scale the position index of the particle.
      xi1 -= ba.xi1_0;
      xi2 -= ba.xi2_0;
      xi3 -= ba.xi3_0;
      if (snb.level > mylevel) {
        xi1 *= 2.0;
        xi2 *= 2.0;
        xi3 *= 2.0;
      } else {
        xi1 *= 0.5;
        xi2 *= 0.5;
        xi3 *= 0.5;
      }

      // Assign the particle.
      Real factor(1.0);
      if (mul_factor) factor = property[tid(k)];
      if (delta_f_enable) factor *= weight_delta_f(k);
#pragma ivdep
#pragma loop count(NPC)
      for (int ix3 = std::max(static_cast<int>(xi3 - dxi3_), 0);
           ix3 <=
           std::min(static_cast<int>(xi3 + dxi3_), destination.GetDim3() - 1);
           ++ix3) {
#pragma loop count(NPC)
        for (int ix2 = std::max(static_cast<int>(xi2 - dxi2_), 0);
             ix2 <=
             std::min(static_cast<int>(xi2 + dxi2_), destination.GetDim2() - 1);
             ++ix2) {
#pragma loop count(NPC)
          for (int ix1 = std::max(static_cast<int>(xi1 - dxi1_), 0);
               ix1 <= std::min(static_cast<int>(xi1 + dxi1_),
                               destination.GetDim1() - 1);
               ++ix1) {
            Real w((active1_ ? _WeightFunction(ix1 + 0.5 - xi1) : 1.0) *
                   (active2_ ? _WeightFunction(ix2 + 0.5 - xi2) : 1.0) *
                   (active3_ ? _WeightFunction(ix3 + 0.5 - xi3) : 1.0) *
                   factor);

            dest_weight(ix3, ix2, ix1) += w;
#pragma loop count(InfoParticleMesh::nprop_aux)
            for (int n = 0; n < nprop; ++n)
              destination(ma1 + n, ix3, ix2, ix1) += w * par(p1 + n, k);
          }
        }
      }
    }
  }
  dest_weight.DeleteAthenaArray();
  return;
}

namespace {
//--------------------------------------------------------------------------------------
//! \fn Real _WeightFunction(Real dxi)
//  \brief evaluates the weight function given index distance.

inline Real _WeightFunction(Real dxi) {
  dxi = std::min(std::abs(dxi), boundary_at_weightFunction);
  return dxi < 0.5 ? 0.75 - dxi * dxi : 0.5 * ((1.5 - dxi) * (1.5 - dxi));
}
}  // namespace