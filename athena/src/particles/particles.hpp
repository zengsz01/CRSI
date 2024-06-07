#ifndef PARTICLES_PARTICLES_HPP_
#define PARTICLES_PARTICLES_HPP_
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors
//======================================================================================
//! \file particles.hpp
//  \brief defines classes for particle dynamics.
//======================================================================================

// C/C++ Standard Libraries
#include <array>
#include <cmath>  // std::sqrt, std::floor
#include <string>
#include <vector>
#include <functional>

// Athena headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/particle/bvals_par.hpp"
#include "../mesh/mesh.hpp"
#include "../outputs/outputs.hpp"
#include "particle-mesh.hpp"

 // number of variables in history output
#define NHISTORY_PAR 7

// Forward declarations
class ParameterInput;

//--------------------------------------------------------------------------------------
//! \class Particles
//  \brief defines the base class for all implementations of particles.

class Particles {
  friend class MeshBlock;  // Make writing initial conditions possible.
  friend class OutputType;
  friend class ATHDF5Output;
  friend class ParticleMesh;
  friend class BoundaryValues;
  friend class ParBoundaryVariable;
  friend BValParFunc;

 public:
  // Class methods
  static void AMRCoarseToFineSameRank(MeshBlock *pmbc, MeshBlock *pmbf);
  static void AMRFineToCoarseSameRank(MeshBlock *pmbf, MeshBlock *pmbc);
  static void RecordIDMax(Mesh *pm);
  static void SetIDMax(Mesh *pm);
  static int PrepareSendSameLevelOrF2CAMR(MeshBlock *pmb, char *&sendbuf);
  static int PrepareSendC2FAMR(MeshBlock *pmb, char *&sendbuf,
                               const LogicalLocation &lloc);
  static void FinishRecvSameLevel(MeshBlock *pmb, const int &size_char,
                                  char *recvbuf);
  static void FinishRecvAMR(MeshBlock *pmb, const int &size_char,
                               char *recvbuf);
  static void FinishRecvC2FAMR(MeshBlock *pmb, const int &size_char,
                               char *recvbuf);
  static void Initialize(Mesh *pm, ParameterInput *pin);
  static void PostInitialize(Mesh *pm, ParameterInput *pin);
  static void FindHistoryOutput(Mesh *pm, Real data_sum[], int pos);
  static void FormattedTableOutput(Mesh *pm, OutputParameters op);
  static void BinaryOutput(Mesh *pm, OutputParameters op);
  static void GetHistoryOutputNames(std::string output_names[]);
  static void GetNumberDensityOnMesh(Mesh *pm, bool include_velocity);
  static int GetTotalNumber(Mesh *pm);
  static void EnrollUserBoundaryFunction(BoundaryFace dir, BValParFunc my_bc);

  void AnalysisOutput(Real *spec, const int &ntbin, const int &npbin,
                              const Real &dpbin_inv,
                              const Real &ln_p_min) const;
  
  // Constructor
  Particles(MeshBlock *pmb, ParameterInput *pin);

  // Destructor
  virtual ~Particles();

  // Accessor
  Real GetMaximumWeight() const;
  virtual std::size_t GetTotalTypeNumber() const;
  int GetLocalNumber() const { return npar; };
  static bool DeltafEnable();
  static bool BackReactionEnable();
  static bool AssignWeightEnable();
  static int Index_X_Vel();
  static int Index_Y_Vel();
  static int Index_Z_Vel();

  // Instance methods
  void ClearParBoundary(BoundaryCommSubset phase);
  virtual void Integrate(int step);
  void Sort();
  void LinkNeighbors();
  void InitRecvPar(BoundaryCommSubset phase);
  void SetParMeshBoundaryAttributes();
  void SendParticleMesh();
  void ReceiveAndSetParMeshWithWait();
  void ReceiveAndSetParWithWait();
  void SendParticles();
  void SetPositionIndices();
  bool ReceiveNumParAndStartRecv();
  bool ReceiveParticles();
  bool ReceiveParticleMesh();
  virtual void SetParticleMesh(int step);
  void SetParticles();
  virtual Real NewBlockTimeStep() const;
  virtual Real Statistics(std::function<Real(Real, Real)> const &func)
      const;  // suppose required statistical value depends on two properties
  void UserWorkInLoop();  // similar to MeshBlock::UserWorkInLoop()
  virtual std::vector<Real> GetTypes() const = 0;
  virtual Real GetTypes(const int &k) const = 0;

  size_t GetSizeInBytes();
  void UnpackParticlesForRestart(char *mbdata, std::size_t &os);
  void PackParticlesForRestart(char *&pdata);

  // Class variable
  static AthenaArray<int> idmax_array;

 protected:
  // Class methods
  static int AddIntProperty();
  static int AddRealProperty();
  static int AddAuxProperty();
  static int AddWorkingArray();
  void GetPositionIndices(int npar, const AthenaArray<Real> &xp,
                          const AthenaArray<Real> &yp,
                          const AthenaArray<Real> &zp, AthenaArray<Real> &xi1,
                          AthenaArray<Real> &xi2, AthenaArray<Real> &xi3);
  virtual void ApplyLogicalConditions(
      const int &indice_for_new);  // After setting particles

  // Class variables
  static bool initialized;  // whether or not the class is initialized
  static int nint, nreal;   // numbers of integer and real particle properties
  static int naux;          // number of auxiliary particle properties
  static int nwork;         // number of working arrays for particles

  static int ipid, iinit_mbid;  // index for the particle ID or the initial meshblock ID
  static int ixp, iyp, izp;     // indices for the position components
  static int ivpx, ivpy, ivpz;  // indices for the velocity components

  static int ixp0, iyp0, izp0;      // indices for beginning position components
  static int ivpx0, ivpy0, ivpz0;   // indices for beginning velocity components
  static int iep0;                  // indices for beginning energy components
  static int if0, idelta_f_weight;  // indices for beginning inv_f0 components
  static int itid;                 // indices for particle property ID (for charged)

  static int ixi1, ixi2, ixi3;  // indices for position indices

  static int imvpx, imvpy, imvpz;  // indices for velocity components on mesh
  static int imep;                 // indices for energy components on mesh

  static Real cfl_par;         // CFL number for particles
  static bool delta_f_enable;  // flagging if the delta f method is needed
  static bool backreaction;    // on/off of back reaction, modified in the derived class

  Real deposit_var[InfoParticleMesh::nprop_aux][SIMD_WIDTH] __attribute__((
      aligned(CACHELINE_BYTES)));  // Variables that need to be deposited
  Real bkgnd[InfoParticleMesh::nprop_bg][SIMD_WIDTH] __attribute__((
      aligned(CACHELINE_BYTES)));  // Variables that need to be interpolated

  // Instance methods
  virtual void AssignShorthands();  // Needs to be called everytime
                                    // intprop, realprop, & auxprop are resized
                                    // Be sure to call back when derived.

  void UpdateCapacity(int new_nparmax);  // Change the capacity of particle arrays
  void SaveStatus();

  // Instance variables
  int npar;                        // number of particles
  int nparmax;                     // maximum
  bool active1, active2, active3;  // active dimensions

  // Data attached to the particles:
  AthenaArray<int> intprop;    //   integer properties
  AthenaArray<Real> realprop;  //   real properties
  AthenaArray<Real> auxprop;   //   auxiliary properties (communicated when
                               //     particles moving to another meshblock)
  AthenaArray<Real> work;      //   working arrays (not communicated)

  ParticleMesh *ppm;  // ptr to particle-mesh

  // Shorthands:
  AthenaArray<int> pid, init_mbid;     //   particle ID, initial meshblock ID
  AthenaArray<int> tid;                //   particle property ID (for charged particles)
  AthenaArray<Real> xp, yp, zp;        //   position
  AthenaArray<Real> vpx, vpy, vpz;     //   velocity
  AthenaArray<Real> xi1, xi2, xi3;     //   position indices in local meshblock
  AthenaArray<Real> xp0, yp0, zp0;     //   beginning position
  AthenaArray<Real> vpx0, vpy0, vpz0;  //   beginning velocity
  AthenaArray<Real> ep0;               //   beginning energy
  AthenaArray<Real> inv_f0;            //   beginning phase space distribution
  AthenaArray<Real> delta_f_weight;    //   copy of delta f, for refinement

  AthenaArray<Real> dpx1_, dpx2_, dpx3_;  // shorthand for momentum change
  AthenaArray<Real> dpe_;                 // shorthand for energy change

  // Instance variables for dust particles
  AthenaArray<Real> wx, wy, wz;        // shorthand for working arrays

  // Instance variables for charged particles
  AthenaArray<Real> field_;               // backgound field copy from meshblock

  MeshBlock *pmy_block;  // MeshBlock pointer
  Mesh *pmy_mesh;        // Mesh pointer

 private:
  // Class method
  static void ProcessNewParticles(Mesh *pmesh);

  // Instance methods
  virtual void SourceTerms(Real t, Real dt,
                           const AthenaArray<Real> &meshsrc) = 0;
  virtual void UserSourceTerms(Real t, Real dt,
                               const AthenaArray<Real> &meshsrc) = 0;
  virtual void ReactToMeshAux(Real t, Real dt,
                              const AthenaArray<Real> &meshsrc) = 0;
  virtual void DepositToMesh(const int &stage, Real t, Real dt,
                             const AthenaArray<Real> &meshsrc,
                             AthenaArray<Real> &meshdst) = 0;
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) \
    linear(ref(x, y, z, px, py, pz))
  virtual Real PhaseDist(const Real &x, const Real &y, const Real &z,
                         const Real &px, const Real &py,
                         const Real &pz) const = 0;
  void RemoveOneParticle(int k);
  int CountNewParticles() const;
  void ApplyBoundaryConditions();  // Before sending particles
  void DispatchBoundaryFunctions(MeshBlock *pmb, Coordinates *pco, Real time,
                                 Real dt, BoundaryFace face);
  void EulerStep(Real t, Real dt, const AthenaArray<Real> &meshsrc);
  //void SetNewParticleID(int id);
  void SetNewParticleID();

  // functions for physical boundaries, called in DispatchBoundaryFunctions
  // reflecting
  void ReflectInnerX1(const Real &time, const Real &dt);
  void ReflectOuterX1(const Real &time, const Real &dt);
  void ReflectInnerX2(const Real &time, const Real &dt);
  void ReflectOuterX2(const Real &time, const Real &dt);
  void ReflectInnerX3(const Real &time, const Real &dt);
  void ReflectOuterX3(const Real &time, const Real &dt);

  // outflow
  void OutflowInnerX1(const Real &time, const Real &dt);
  void OutflowOuterX1(const Real &time, const Real &dt);
  void OutflowInnerX2(const Real &time, const Real &dt);
  void OutflowOuterX2(const Real &time, const Real &dt);
  void OutflowInnerX3(const Real &time, const Real &dt);
  void OutflowOuterX3(const Real &time, const Real &dt);
  
  // polar wedge: No differences between outter and inner
  void PolarWedgeX2(const Real &time, const Real &dt);

  // Class variable
  int idmax;

  // MeshBlock-to-MeshBlock communication:
  ParBoundaryVariable pbvar_;        // boundary communicate variable
  // ptr to my BoundaryValues, using for physical and logical boundaries
  static BValParFunc BoundaryFunction_[6];
};


//--------------------------------------------------------------------------------------
//! \fn Real Particles::GetMaximumWeight()
//  \brief returns the maximum weight on the mesh.

inline Real Particles::GetMaximumWeight() const {
  return ppm->FindMaximumWeight();
}

inline std::size_t Particles::GetTotalTypeNumber() const { return 1; }

//--------------------------------------------------------------------------------------
//! \class DustParticles
//  \brief defines the class for dust particles that interact with the gas via
//  drag
//         force.

class DustParticles : public Particles {
  friend class MeshBlock;

 public:
  // Class method
  static void Initialize(Mesh *pm, ParameterInput *pin);
  static void SetOneParticleMass(Real new_mass);
  static Real GetOneParticleMass();
  static Real GetStoppingTime();

  // Constructor
  DustParticles(MeshBlock *pmb, ParameterInput *pin);

  // Destructor
  ~DustParticles();

  // Instance method
  Real NewBlockTimeStep() const override;
  std::vector<Real> GetTypes() const override;
  Real GetTypes(const int &k) const override;

 protected:
  static Real mass;          // mass of each particle
  static Real taus;          // stopping time (in code units)

 private:
  // Class variables
  static bool initialized;         // whether or not the class is initialized
  static int iwx, iwy, iwz;        // indices for working arrays

  // Instance methods.
  void SourceTerms(Real t, Real dt, const AthenaArray<Real> &meshsrc) override;
  void UserSourceTerms(Real t, Real dt,
                       const AthenaArray<Real> &meshsrc) override;
  void DepositToMesh(const int &stage, Real t, Real dt,
                     const AthenaArray<Real> &meshsrc,
                     AthenaArray<Real> &meshdst) override;
  void ReactToMeshAux(Real t, Real dt,
                      const AthenaArray<Real> &meshsrc) override;
  Real PhaseDist(const Real &x, const Real &y, const Real &z, const Real &px,
                 const Real &py, const Real &pz) const override;
};

//--------------------------------------------------------------------------------------
//! \fn void DustParticles::SetOneParticleMass(Real new_mass)
//  \brief sets the mass of each particle.

inline void DustParticles::SetOneParticleMass(Real new_mass) {
  mass = new_mass;
}

//--------------------------------------------------------------------------------------
//! \fn Real DustParticles::GetOneParticleMass()
//  \brief returns the mass of each particle.

inline Real DustParticles::GetOneParticleMass() { return mass; }

//--------------------------------------------------------------------------------------
//! \fn Real DustParticles::GetStoppingTime()
//  \brief returns the stopping time of the drag.

inline Real DustParticles::GetStoppingTime() { return taus; }

//--------------------------------------------------------------------------------------
//! \class ChargedParticles
//  \brief defines a class for charged particles interacting with mhd through
//         eletric currents and electromagnetic force.

//   The velocity from the base class Particles(ie. vpx, vpx0) stands for the
//   four velcity in corresponding directions(ie. vpx = \gamma ux, ux = vpx /
//   \sqrt((vpx^2 + vpy^2 + vpz^2)/c_^2 + 1)

class ChargedParticles : public Particles {
  friend class MeshBlock;

 public:
  // Class method
  static void Initialize(Mesh *pm, ParameterInput *pin);
  static void SetOneParticleMass(const std::size_t &tid, const Real &new_mass);
  static void SetOneParticlecharge(const std::size_t &tid, const Real &new_q);
  static void SetSpeedOfLight(Real new_c);
  static std::vector<Real> GetOneParticleMass();
  static std::vector<Real> GetOneParticlecharge();
  static Real GetSpeedOfLight();
  std::size_t GetTotalTypeNumber() const override;

  // Constructor
  ChargedParticles(MeshBlock *pmb, ParameterInput *pin);

  // Destructor
  ~ChargedParticles();

  // Instance method
  Real NewBlockTimeStep() const override;
  Real Statistics(std::function<Real(Real, Real)> const &func) const override;
  void SetParticleMesh(int step) override;
  void Integrate(int stage) override;
  std::vector<Real> GetTypes() const override;
  Real GetTypes(const int &k) const override;
  
 private:
  // Class variables
  static std::vector<Real> mass_;  // mass of each particle
  static std::vector<Real> q_;     // charge of each particle
  static Real c_;                  // speed of light, only used for particles at present

  // Class methods.

  // TODO(sxc18): implement for subcycling

  // Instance methods.
  void UserSourceTerms(Real t, Real dt,
                       const AthenaArray<Real> &meshsrc) override;
  void DepositToMesh(const int &stage, Real t, Real dt,
                     const AthenaArray<Real> &meshsrc,
                     AthenaArray<Real> &meshdst) override;
  void SourceTerms(Real t, Real dt, const AthenaArray<Real> &meshsrc) override;
  void ReactToMeshAux(Real t, Real dt,
                      const AthenaArray<Real> &meshsrc) override;
  void AssignCurrent(const int &stage, Real t, Real dt);
  Real PhaseDist(const Real &x, const Real &y, const Real &z, const Real &px,
                 const Real &py, const Real &pz) const override;
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) linear(ref(x, y, z))
  Real ChargeDen0(const Real &x, const Real &y, const Real &z);
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) linear(ref(x, y, z))
  Real Jx0(const Real &x, const Real &y, const Real &z);
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) linear(ref(x, y, z))
  Real Jy0(const Real &x, const Real &y, const Real &z);
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) linear(ref(x, y, z))
  Real Jz0(const Real &x, const Real &y, const Real &z);
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) linear(ref(x, y, z))
  Real DeltaMomx0(const Real &x, const Real &y, const Real &z);
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) linear(ref(x, y, z))
  Real DeltaMomy0(const Real &x, const Real &y, const Real &z);
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) linear(ref(x, y, z))
  Real DeltaMomz0(const Real &x, const Real &y, const Real &z);
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this) linear(ref(x, y, z))
  Real DeltaEnergy0(const Real &x, const Real &y, const Real &z);

  void XformFdbk2MeshCoord();
  void ApplyLogicalConditions(
      const int &indice_for_new) override;  // apply phase randomization

  // Instance variables
  static bool initialized_;           // whether or not the class is initialized
  static Real cfl_rot_;                // CFL number for particles
  static bool phase_random_;         // whether or not to randomize phase
};

//--------------------------------------------------------------------------------------
//! \fn void ChargedParticles::SetOneParticleMass(const std::size_t &tid, const
//! Real &new_mass)
//  \brief sets the mass of each particle.

inline void ChargedParticles::SetOneParticleMass(const std::size_t &tid,
                                                 const Real &new_mass) {
  if (tid < mass_.size()) mass_[tid] = new_mass;
}

//--------------------------------------------------------------------------------------
//! \fn void ChargedParticles::SetOneParticlecharge(const std::size_t &tid,
//! const Real &new_q)
//  \brief sets the electric charge of each particle.

inline void ChargedParticles::SetOneParticlecharge(const std::size_t &tid,
                                                   const Real &new_q) {
  if (tid < q_.size()) q_[tid] = new_q;
}

//--------------------------------------------------------------------------------------
//! \fn void ChargedParticles::SetOneParticlecharge(Real new_q)
//  \brief sets the speed of light in the particle module.

inline void ChargedParticles::SetSpeedOfLight(Real new_c) { c_ = new_c; }

//--------------------------------------------------------------------------------------
//! \fn std::vector<Real> ChargedParticles::GetOneParticleMass()
//  \brief returns the mass of each particle.

inline std::vector<Real> ChargedParticles::GetOneParticleMass() {
  return mass_;
}

//--------------------------------------------------------------------------------------
//! \fn std::vector<Real> ChargedParticles::GetOneParticlecharge()
//  \brief returns ithe electric charge of each particle.

inline std::vector<Real> ChargedParticles::GetOneParticlecharge() { return q_; }

//--------------------------------------------------------------------------------------
//! \fn Real ChargedParticles::GetSpeedOfLight()
//  \brief returns the speed of light in the particle module.

inline Real ChargedParticles::GetSpeedOfLight() { return c_; }

inline std::size_t ChargedParticles::GetTotalTypeNumber() const { 
  if (mass_.size() == q_.size())
    return mass_.size();
  else
    return 0;
}

#endif  // PARTICLES_PARTICLES_HPP_