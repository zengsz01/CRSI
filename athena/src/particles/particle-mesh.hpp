#ifndef PARTICLES_PARTICLE_MESH_HPP_
#define PARTICLES_PARTICLE_MESH_HPP_
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//======================================================================================
//! \file particle-mesh.hpp
//  \brief defines ParticleMesh class used for communication between meshblocks
//  needed by particle-mesh methods.

// C++ standard library
#include <cmath>

// Athena++ classes headers
#include "../athena_arrays.hpp"
#include "../bvals/particle/bvals_par_mesh.hpp"
#include "../mesh/mesh.hpp"

// Particle-mesh constants.
const Real RINF = 1;  // radius of influence

// Define the size of a particle cloud = 2 * RINF + 1
#define NPC 3
#define NPC_SQR 9    // NPC * NPC
#define NPC_CUBE 27  // NPC * NPC * NPC

// Forward declaration
class Particles;
class ParameterInput;

namespace InfoParticleMesh {
// Total number of elements in the background_3d_ (to be interpolated)
#if PARTICLES == CHARGED_PAR
constexpr int nprop_bg(6);
#else
constexpr int nprop_bg(3);
#endif

// array indices for the backgournd
enum BckgdIndex { IVx = 0, IVy = 1, IVz = 2, IBx = 3, IBy = 4, IBz = 5 };

// Total number of elements in the auxiliary (to be deposited)
#if NON_BAROTROPIC_EOS
constexpr int nprop_aux(4);
constexpr int nprop_aux_plus_one(5);
#else
constexpr int nprop_aux(3);
constexpr int nprop_aux_plus_one(4);
#endif  // NON_BAROTROPIC_EOS

// array indices for the auxiliary
enum AuxIndex { IPx = 0, IPy = 1, IPz = 2, IE = 3 };

typedef Real IntermediateBackground3D[NPC_SQR][nprop_bg]
    __attribute__((aligned(CACHELINE_BYTES)));
typedef Real IntermediateAuxiliary3D[NPC_SQR][nprop_aux_plus_one]
    __attribute__((aligned(CACHELINE_BYTES)));
typedef Real IntermediateBackground2D[NPC][nprop_bg]
    __attribute__((aligned(CACHELINE_BYTES)));
typedef Real IntermediateAuxiliary2D[NPC][nprop_aux_plus_one]
    __attribute__((aligned(CACHELINE_BYTES)));
typedef Real IntermediateBackground1D[nprop_bg]
    __attribute__((aligned(CACHELINE_BYTES)));
typedef Real IntermediateAuxiliary1D[nprop_aux_plus_one]
    __attribute__((aligned(CACHELINE_BYTES)));
}  // namespace InfoParticleMesh

//--------------------------------------------------------------------------------------
//! \class ParticleMesh
//  \brief defines the class for particle-mesh methods

class ParticleMesh {
  friend class Particles;
  friend class DustParticles;
  friend class ChargedParticles;
  friend class OutputType;

 public:
  // Class methods
  static void Initialize(Mesh *pm, ParameterInput *pin);
  static void AddWeight();
  static int AddMeshAux();
  void InitIntermAux(const Real &value = 0.0);
  void InitIntermBkgnd3D(const AthenaArray<Real> &meshsrc);
  void InitIntermBkgnd2D(const AthenaArray<Real> &meshsrc);
  void InitIntermBkgnd1D(const AthenaArray<Real> &meshsrc);
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) linear(ref(simd_loc))
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) \
    linear(ref(xi1, xi2, xi3))
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) \
    linear(ref(xi1, xi2, xi3, simd_loc))
  void GetWeightTSC(const Real &xi1, const Real &xi2, const Real &xi3,
                    const int &simd_loc);
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) linear(ref(simd_loc))
  void DeltaFWeight(const Real &factor, const int &simd_loc) {
    w1_[0][simd_loc] *= factor;
    w1_[1][simd_loc] *= factor;
    w1_[2][simd_loc] *= factor;
  };
  void Interpolate4IntermBkgnd3D(const int &simd_loc);
  void Interpolate4IntermBkgnd2D(const int &simd_loc);
  void Interpolate4IntermBkgnd1D(const int &simd_loc);
  void Assign2IntermAux3D(const int &simd_loc,
                          const int &nprop = InfoParticleMesh::nprop_aux);
  void Assign2IntermAux2D(const int &simd_loc,
                          const int &nprop = InfoParticleMesh::nprop_aux);
  void Assign2IntermAux1D(const int &simd_loc,
                          const int &nprop = InfoParticleMesh::nprop_aux);
  void Assign2IntermAuxAndWght3D(const int &simd_loc,
                        const int &nprop = InfoParticleMesh::nprop_aux);
  void Assign2IntermAuxAndWght2D(
      const int &simd_loc, const int &nprop = InfoParticleMesh::nprop_aux);
  void Assign2IntermAuxAndWght1D(
      const int &simd_loc, const int &nprop = InfoParticleMesh::nprop_aux);

  // Constructor and destructor
  explicit ParticleMesh(Particles *ppar);
  ~ParticleMesh();

  // Accessor
  Real FindMaximumWeight() const;
  int GetCellNumDim1() const;
  int GetCellNumDim2() const;
  int GetCellNumDim3() const;
  int GetTotalCellNum() const;

 protected:
  // Class variables
  static int nmeshaux;  // number of auxiliaries to the meshblock
  static int iweight;   // index to weight in meshaux

  // Instance variables
  AthenaArray<Real> meshaux;   // auxiliaries to the meshblock
  int is, ie, js, je, ks, ke;  // beginning and ending indices
  AthenaArray<Real> weight;    // shorthand to weight in meshaux
  AthenaArray<Real> vol;       // volume of each cell (in DepositMeshAux)

  // Instance methods
  void InterpolateMeshToParticles(const AthenaArray<Real> &meshsrc, int ms1,
                                  AthenaArray<Real> &par, int p1, int nprop);
  void AssignParticlesToMeshAux(const AthenaArray<Real> &par, int p1, int ma1,
                                int nprop);
  void InterpolateMeshAndAssignParticles(const AthenaArray<Real> &meshsrc,
                                         int ms1, AthenaArray<Real> &pardst,
                                         int pd1, int ni,
                                         const AthenaArray<Real> &parsrc,
                                         int ps1, int ma1, int na);
  void AssignIntermAux3D2MeshAux(const int &ma, const int &begin,
                               const int &nprop, const int &last,
                               const bool &flag_rho);
  void AssignIntermAux2D2MeshAux(const int &ma, const int &begin,
                                 const int &nprop, const int &last,
                                 const bool &flag_rho);
  void AssignIntermAux1D2MeshAux(const int &ma, const int &begin,
                                 const int &nprop, const int &last,
                                 const bool &flag_rho);
  void DepositMeshAux(AthenaArray<Real> &u, int ma1, int mb1, int nprop);

 private:
  struct BoundaryAttributes {
    Real xi1min, xi1max, xi2min, xi2max, xi3min, xi3max;
    // domain that influences the ghost block
    Real xi1_0, xi2_0, xi3_0;
    // origin of the ghost block wrt to the local meshblock
  };

  // Class variables
  static bool initialized_;

  // Instance Variables
  static bool active1_, active2_, active3_;  // active dimensions
  static Real dxi1_, dxi2_, dxi3_;  // range of influence from a particle cloud
  int nx1_, nx2_, nx3_;      // number of cells in meshaux in each dimension
  int ncells_;               // total number of cells in meshaux
  static int npc1_, npc2_, npc3_;   // size of a particle cloud

  // Background field information to interpolate
  InfoParticleMesh::IntermediateBackground3D *background_3d_;
  InfoParticleMesh::IntermediateBackground2D *background_2d_;
  InfoParticleMesh::IntermediateBackground1D *background_1d_;
  // Feedback information to assign
  InfoParticleMesh::IntermediateAuxiliary3D *interm_aux_3d_;
  InfoParticleMesh::IntermediateAuxiliary2D *interm_aux_2d_;
  InfoParticleMesh::IntermediateAuxiliary1D *interm_aux_1d_;

  // weight during interpolation or deposit
  Real w1_[NPC][SIMD_WIDTH]
      __attribute__((aligned(CACHELINE_BYTES)));  // x- dir
  Real w_slice_[NPC_SQR][SIMD_WIDTH]
      __attribute__((aligned(CACHELINE_BYTES)));  // y- and z- dir
  int ixv_[SIMD_WIDTH] __attribute__((
      aligned(CACHELINE_BYTES)));  // the location of the particle on the
                                   // meshblock, in 1d.

  // Code of the neighbouring position to be interpolated or deposited
  static int ipc1v_[NPC_CUBE] __attribute__((aligned(CACHELINE_BYTES)));
  static int ipc2v_[NPC_CUBE] __attribute__((aligned(CACHELINE_BYTES)));
  static int ipc3v_[NPC_CUBE] __attribute__((aligned(CACHELINE_BYTES)));
  static int islice_[NPC_CUBE] __attribute__((aligned(CACHELINE_BYTES)));
  static int ipc2v_slice_[NPC_SQR] __attribute__((aligned(CACHELINE_BYTES)));
  static int ipc3v_slice_[NPC_SQR] __attribute__((aligned(CACHELINE_BYTES)));
  // Number of cells in a particle cloud, equal to 27 in 3d simulation, 9 in 2d,
  // 3 in 1d when TSC
  static int ncloud_;
  // Number of cells in a sliced particle cloud, equal to 9 in 3d simulation, 3
  // in 2d, 1 in 1d when TSC
  static int ncloud_slice_;
  int nx1_eff_, nx2_eff_, nx3_eff_;  // Effective cell size

  Particles *ppar_;            // ptr to my Particles instance
  MeshBlock *pmb_;             // ptr to my MeshBlock
  Mesh *pmesh_;                // ptr to my Mesh
  BoundaryAttributes ba_[56];  // ghost block attributes
  ParMeshBoundaryVariable pmbvar_; // boundary communicate variable

  // Instance methods
  void SetBoundaryAttributes();
  void AssignParticlesToDifferentLevels(
      const AthenaArray<Real> &par, int p1, int ma1, int nprop,
      const AthenaArray<int> &tid = AthenaArray<int>(),
      const std::vector<Real> &property = std::vector<Real>(),
      const AthenaArray<Real> &weight_delta_f = AthenaArray<Real>());
#pragma omp declare simd simdlen(SIMD_WIDTH) \
    uniform(this, ix_tmp, islice_tmp, type_tmp)
#pragma omp declare simd simdlen(SIMD_WIDTH) \
    uniform(this, islice_tmp, type_tmp)
  inline int GetIndexIntermBkgnd(const int &ix_tmp, const int &islice_tmp,
                                 const int &type_tmp) const {
    return (type_tmp +
            InfoParticleMesh::nprop_bg * (islice_tmp + ncloud_slice_ * ix_tmp));
  };
#pragma omp declare simd simdlen(SIMD_WIDTH) \
    uniform(this, ix_tmp, type_tmp) linear(ref(islice_tmp))
#pragma omp declare simd simdlen(SIMD_WIDTH) \
    uniform(this, type_tmp) linear(ref(ix_tmp, islice_tmp))
  inline int GetIndexIntermAux(const int &ix_tmp, const int &islice_tmp,
                               const int &type_tmp) const {
    return (type_tmp + InfoParticleMesh::nprop_aux_plus_one *
                           (islice_tmp + ncloud_slice_ * ix_tmp));
  };
};
#endif  // PARTICLES_PARTICLE_MESH_HPP_
