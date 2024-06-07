//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file particles.cpp
//  \brief implements functions in particle classes

// C++ Standard Libraries
#include <algorithm>
#include <cmath>     // std::log2
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>     // std::memcpy
#include <vector>
#include <iostream>   // std::endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <functional>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/meshblock_tree.hpp"
#include "../field/field.hpp"
#include "../bvals/bvals.hpp"
#include "../bvals/particle/bvals_par_mesh.hpp"
#include "../utils/utils.hpp"
#include "particles.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// Class variable initialization
bool Particles::initialized = false;
bool Particles::backreaction = false;
bool Particles::delta_f_enable = false;
int Particles::nint = 0;
int Particles::nreal = 0;
int Particles::naux = 0;
int Particles::nwork = 0;
int Particles::ipid = -1, Particles::iinit_mbid = -1;
int Particles::ixp = -1, Particles::iyp = -1, Particles::izp = -1;
int Particles::ivpx = -1, Particles::ivpy = -1, Particles::ivpz = -1;
int Particles::ixp0 = -1, Particles::iyp0 = -1, Particles::izp0 = -1;
int Particles::ivpx0 = -1, Particles::ivpy0 = -1, Particles::ivpz0 = -1;
int Particles::iep0 = -1;
int Particles::if0 = -1, Particles::idelta_f_weight = -1;
int Particles::ixi1 = -1, Particles::ixi2 = -1, Particles::ixi3 = -1;
int Particles::imvpx = -1, Particles::imvpy = -1, Particles::imvpz = -1;
int Particles::imep = -1;
int Particles::itid = -1;
Real Particles::cfl_par = 1;
AthenaArray<int> Particles::idmax_array = AthenaArray<int>();
BValParFunc Particles::BoundaryFunction_[6] = {nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr};


namespace {
// Total number of elements to be output in Particles::BinaryOutput
constexpr int nbuf_box(12);
constexpr int nbuf_time(2);
constexpr int nbuf_type_max(3);
constexpr int size_BufferPar(8 * sizeof(float) + sizeof(long) + sizeof(int));

// Indices of elements to be output in Particles::BinaryOutput
enum BufBoxIndex {
  ImbXMin = 0,
  ImbXMax = 1,
  ImbYMin = 2,
  ImbYMax = 3,
  ImbZMin = 4,
  ImbZMax = 5,
  ImXMin = 6,
  ImXMax = 7,
  ImYMin = 8,
  ImYMax = 9,
  ImZMin = 10,
  ImZMax = 11,
};
enum BufTimeIndex { ITime = 0, Idt = 1 };


struct BufferPar {
  float xp, yp, zp, vpx, vpy, vpz, dpar, property;
  //int property;
  long pid;
  int init_mbid;
};
typedef float BufferBox[nbuf_box];
typedef float BufferTime[nbuf_time];
typedef float BufferType[nbuf_type_max];
}  // namespace

//--------------------------------------------------------------------------------------
//! \fn void Particles::AMRCoarseToFineSameRank(MeshBlock* pmbc, MeshBlock* pmbf)
//  \brief load particles from a coarse meshblock to a fine meshblock.

void Particles::AMRCoarseToFineSameRank(MeshBlock* pmbc, MeshBlock* pmbf) {
  // Initialization
  Particles *pparc = pmbc->ppar, *pparf = pmbf->ppar;
  const Real x1min = pmbf->block_size.x1min, x1max = pmbf->block_size.x1max;
  const Real x2min = pmbf->block_size.x2min, x2max = pmbf->block_size.x2max;
  const Real x3min = pmbf->block_size.x3min, x3max = pmbf->block_size.x3max;
  const bool active1 = pparc->active1,
             active2 = pparc->active2,
             active3 = pparc->active3;
  const AthenaArray<Real> &xp = pparc->xp, &yp = pparc->yp, &zp = pparc->zp;
  const Coordinates *pcoord = pmbf->pcoord;

  // Loop over particles in the coarse meshblock.
  for (int k = 0; k < pparc->npar; ++k) {
    Real x1, x2, x3;
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if ((!active1 || (active1 && x1min <= x1 && x1 < x1max)) &&
        (!active2 || (active2 && x2min <= x2 && x2 < x2max)) &&
        (!active3 || (active3 && x3min <= x3 && x3 < x3max))) {
      // Load a particle to the fine meshblock.
      int npar = pparf->npar;
      if (npar >= pparf->nparmax) pparf->UpdateCapacity(2 * pparf->nparmax);
      for (int j = 0; j < nint; ++j)
        pparf->intprop(j,npar) = pparc->intprop(j,k);
      for (int j = 0; j < nreal; ++j)
        pparf->realprop(j,npar) = pparc->realprop(j,k);
      for (int j = 0; j < naux; ++j)
        pparf->auxprop(j,npar) = pparc->auxprop(j,k);
      ++pparf->npar;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::AMRFineToCoarseSameRank(MeshBlock* pmbf, MeshBlock* pmbc)
//  \brief load particles from a fine meshblock to a coarse meshblock.

void Particles::AMRFineToCoarseSameRank(MeshBlock* pmbf, MeshBlock* pmbc) {
  // Check the capacity.
  Particles *pparf = pmbf->ppar, *pparc = pmbc->ppar;
  int nparf = pparf->npar, nparc = pparc->npar;
  int npar_new = nparf + nparc;
  if (npar_new > pparc->nparmax) pparc->UpdateCapacity(npar_new);

  // Load the particles.
  for (int j = 0; j < nint; ++j)
    for (int k = 0; k < nparf; ++k)
      pparc->intprop(j,nparc+k) = pparf->intprop(j,k);
  for (int j = 0; j < nreal; ++j)
    for (int k = 0; k < nparf; ++k)
      pparc->realprop(j,nparc+k) = pparf->realprop(j,k);
  for (int j = 0; j < naux; ++j)
    for (int k = 0; k < nparf; ++k)
      pparc->auxprop(j,nparc+k) = pparf->auxprop(j,k);
  pparc->npar = npar_new;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::RecordIDMax(Mesh *pm, int *idmax_array, const int &nbtold)
//  \brief record an array containing idmax for each meshblock

void Particles::RecordIDMax(Mesh *pm) {
  for (int i = 0; i < pm->nblocal; ++i)
    idmax_array(pm->my_blocks(i)->gid) = pm->my_blocks(i)->ppar->idmax;
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, idmax_array.data(), pm->nbtotal, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
#endif
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SetIDMax(Mesh *pm, int *idmax_array, const int
//! &nbtold)
//  \brief set idmax from the array and newborn meshblock with idmax = 0

void Particles::SetIDMax(Mesh *pm) {
  if (pm->nbtotal > idmax_array.GetDim1())
    idmax_array.ResizeLastDimension(pm->nbtotal);
  for (int i = 0; i < pm->nblocal; ++i)
    pm->my_blocks(i)->ppar->idmax = idmax_array(pm->my_blocks(i)->gid);
                                        
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::PrepareSendSameLevelOrF2CAMR(MeshBlock *pmb, char *sendbuf)
//  \brief load particles from pmb to the meshblcok in the same level or a
//  coaser one, and return the length of sendbuf

int Particles::PrepareSendSameLevelOrF2CAMR(MeshBlock *pmb, char *&sendbuf) {
  delete[] sendbuf;
  const int size_total((nint * sizeof(int) + (nreal + naux) * sizeof(Real)) *
                 pmb->ppar->npar + 2);
  if (size_total > 0) {
    sendbuf = new char[size_total];
    sendbuf[0] = false;
    sendbuf[1] = false;
    std::size_t os(2);
    // Write integer properties.
    std::size_t size = pmb->ppar->npar * sizeof(int);
    for (int k = 0; k < nint; ++k) {
#pragma ivdep
      std::memcpy(&(sendbuf[os]), &(pmb->ppar->intprop(k, 0)), size);
      os += size;
    }
    // Write real properties.
    size = pmb->ppar->npar * sizeof(Real);
    for (int k = 0; k < nreal; ++k) {
#pragma ivdep
      std::memcpy(&(sendbuf[os]), &(pmb->ppar->realprop(k, 0)), size);
      os += size;
    }
    for (int k = 0; k < naux; ++k) {
#pragma ivdep
      std::memcpy(&(sendbuf[os]), &(pmb->ppar->auxprop(k, 0)), size);
      os += size;
    }
  } else {
    sendbuf = new char[1]; // Avoid possible problem when delete and send
  }
  return size_total;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::PrepareSendC2FAMR(MeshBlock *pmb, char *&sendbuf,  const
//! LogicalLocation &lloc)
//  \brief load particles from pmb to the meshblcok in a finer level, and return the length of sendbuf

int Particles::PrepareSendC2FAMR(MeshBlock *pmb, char *&sendbuf,
                                 const LogicalLocation &lloc) {
  //if (pmb->ppar->npar <= 0) return 0;
  int ox1 = ((lloc.lx1 & 1LL) == 1LL), ox2 = ((lloc.lx2 & 1LL) == 1LL),
      ox3 = ((lloc.lx3 & 1LL) == 1LL);
  const bool active1 = pmb->ppar->active1, active2 = pmb->ppar->active2,
             active3 = pmb->ppar->active3;
  // pack
  int il, iu, jl, ju, kl, ku;
  if (ox1 == 0)
    il = pmb->is - 1, iu = pmb->is + pmb->block_size.nx1 / 2;
  else
    il = pmb->is + pmb->block_size.nx1 / 2, iu = pmb->ie + 1;
  if (ox2 == 0)
    jl = pmb->js - pmb->pmy_mesh->f2, ju = pmb->js + pmb->block_size.nx2 / 2;
  else
    jl = pmb->js + pmb->block_size.nx2 / 2, ju = pmb->je + pmb->pmy_mesh->f2;
  if (ox3 == 0)
    kl = pmb->ks - pmb->pmy_mesh->f3, ku = pmb->ks + pmb->block_size.nx3 / 2;
  else
    kl = pmb->ks + pmb->block_size.nx3 / 2, ku = pmb->ke + pmb->pmy_mesh->f3;

  int **intprop_tmp = new int *[nint], **intprop_cpy = new int *[nint];
  for (int k = 0; k < nint; ++k) {
    intprop_tmp[k] = new int[pmb->ppar->npar];
    intprop_cpy[k] = intprop_tmp[k];
  }
  Real **realprop_tmp = new Real *[nreal], **auxprop_tmp = new Real *[naux],
       **realprop_cpy = new Real *[nreal], **auxprop_cpy = new Real *[naux];
  for (int k = 0; k < nreal; ++k) {
    realprop_tmp[k] = new Real[pmb->ppar->npar];
    realprop_cpy[k] = realprop_tmp[k];
  }
  for (int k = 0; k < naux; ++k) {
    auxprop_tmp[k] = new Real[pmb->ppar->npar];
    auxprop_cpy[k] = auxprop_tmp[k];
  }

  int npar_send(0);
  for (int k = 0; k < pmb->ppar->npar; ++k) {
    if ((!active1 ||
         (active1 && il <= pmb->ppar->xi1(k) && pmb->ppar->xi1(k) < iu)) &&
        (!active2 ||
         (active2 && jl <= pmb->ppar->xi2(k) && pmb->ppar->xi2(k) < ju)) &&
        (!active3 ||
         (active3 && kl <= pmb->ppar->xi3(k) && pmb->ppar->xi3(k) < ku))) {
      ++npar_send;
      for (int n = 0; n < nint; ++n) {
        *intprop_cpy[n] = pmb->ppar->intprop(n, k);
        ++intprop_cpy[n];
      }
      for (int n = 0; n < nreal; ++n) {
        *realprop_cpy[n] = pmb->ppar->realprop(n, k);
        ++realprop_cpy[n];
      }
      for (int n = 0; n < naux; ++n) {
        *auxprop_cpy[n] = pmb->ppar->auxprop(n, k);
        ++auxprop_cpy[n];
      }
    }
  }

  delete[] sendbuf;
  if (npar_send > 0) {
    sendbuf = new char[(nint * sizeof(int) + (nreal + naux) * sizeof(Real)) *
                       npar_send + 2];
    sendbuf[0] = false;
    sendbuf[1] = false;
    std::size_t os(2);
    // Write integer properties.
    std::size_t size = npar_send * sizeof(int);
    for (int k = 0; k < nint; ++k) {
#pragma ivdep
      std::memcpy(&(sendbuf[os]), &intprop_tmp[k][0], size);
      os += size;
    }
    // Write real properties.
    size = npar_send * sizeof(Real);
    for (int k = 0; k < nreal; ++k) {
#pragma ivdep
      std::memcpy(&(sendbuf[os]), &realprop_tmp[k][0], size);
      os += size;
    }
    for (int k = 0; k < naux; ++k) {
#pragma ivdep
      std::memcpy(&(sendbuf[os]), &auxprop_tmp[k][0], size);
      os += size;
    }
  } else {
    sendbuf = new char[2];  // Avoid possible problem when delete and send
    sendbuf[0] = false;
    sendbuf[1] = false;
  }

  for (int k = 0; k < nint; ++k) delete[] intprop_tmp[k];
  delete[] intprop_tmp;
  delete[] intprop_cpy;
  for (int k = 0; k < nreal; ++k) delete[] realprop_tmp[k];
  delete[] realprop_tmp;
  delete[] realprop_cpy;
  for (int k = 0; k < naux; ++k) delete[] auxprop_tmp[k];
  delete[] auxprop_tmp;
  delete[] auxprop_cpy;

  return (nint * sizeof(int) + (nreal + naux) * sizeof(Real)) * npar_send + 2;
}

//--------------------------------------------------------------------------------------
//! \fn Particles::FinishRecvSameLevel(MeshBlock *pmb, const int &size_char, char
//! *recvbuf)
//  \brief load particles from a same level but from a different rank.

void Particles::FinishRecvSameLevel(MeshBlock *pmb, const int &size_char,
                                    char *recvbuf) {
  pmb->ppar->npar =
      size_char / (nint * sizeof(int) + (nreal + naux) * sizeof(Real));
  if (pmb->ppar->nparmax < pmb->ppar->npar)
    pmb->ppar->UpdateCapacity(pmb->ppar->npar);
  if (pmb->ppar->npar > 0) {
    // First copy data from buffer
    std::size_t os(0);
    // Read integer properties.
    std::size_t size = pmb->ppar->npar * sizeof(int);
    for (int k = 0; k < nint; ++k) {
#pragma ivdep
      std::memcpy(&(pmb->ppar->intprop(k, 0)), &(recvbuf[os]), size);
      os += size;
    }

    // Read real properties.
    size = pmb->ppar->npar * sizeof(Real);
    for (int k = 0; k < nreal; ++k) {
#pragma ivdep
      std::memcpy(&(pmb->ppar->realprop(k, 0)), &(recvbuf[os]), size);
      os += size;
    }
    for (int k = 0; k < naux; ++k) {
#pragma ivdep
      std::memcpy(&(pmb->ppar->auxprop(k, 0)), &(recvbuf[os]), size);
      os += size;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn Particles::FinishRecvAMR(MeshBlock *pmb, const int &size_char, char
//! *recvbuf)
//  \brief load particles from a fine meshblock to a coarse meshblock.

void Particles::FinishRecvAMR(MeshBlock *pmb, const int &size_char,
                                 char *recvbuf) {
  const int npar_new =
      (size_char-2) / (nint * sizeof(int) + (nreal + naux) * sizeof(Real));
  if (pmb->ppar->nparmax < (pmb->ppar->npar + npar_new))
    pmb->ppar->UpdateCapacity(pmb->ppar->npar + npar_new);
  if (npar_new > 0) {
    // First copy data from buffer
    std::size_t os(2);
    // Read integer properties.
    std::size_t size = npar_new * sizeof(int);
    for (int k = 0; k < nint; ++k) {
#pragma ivdep
      std::memcpy(&(pmb->ppar->intprop(k, pmb->ppar->npar)), &(recvbuf[os]),
                  size);
      os += size;
    }

    // Read real properties.
    size = npar_new * sizeof(Real);
    for (int k = 0; k < nreal; ++k) {
#pragma ivdep
      std::memcpy(&(pmb->ppar->realprop(k, pmb->ppar->npar)), &(recvbuf[os]),
                  size);
      os += size;
    }
    for (int k = 0; k < naux; ++k) {
#pragma ivdep
      std::memcpy(&(pmb->ppar->auxprop(k, pmb->ppar->npar)), &(recvbuf[os]),
                  size);
      os += size;
    }

    pmb->ppar->npar += npar_new;
  }
}

//--------------------------------------------------------------------------------------
//! \fn Particles::FinishRecvC2FAMR(MeshBlock *pmb, const int &size_char, char
//! *recvbuf)
//  \brief load particles from a coarse meshblock to a fine meshblock.

void Particles::FinishRecvC2FAMR(MeshBlock *pmb, const int &size_char,
                                 char *recvbuf) {
  pmb->ppar->npar =
      size_char / (nint * sizeof(int) + (nreal + naux) * sizeof(Real));
  if (pmb->ppar->nparmax < pmb->ppar->npar)
    pmb->ppar->UpdateCapacity(pmb->ppar->npar);
  if (pmb->ppar->npar > 0) {
    // First copy data from buffer
    std::size_t os(0);
    // Read integer properties.
    std::size_t size = pmb->ppar->npar * sizeof(int);
    for (int k = 0; k < nint; ++k) {
#pragma ivdep
      std::memcpy(&(pmb->ppar->intprop(k, 0)), &(recvbuf[os]), size);
      os += size;
    }

    // Read real properties.
    size = pmb->ppar->npar * sizeof(Real);
    for (int k = 0; k < nreal; ++k) {
#pragma ivdep
      std::memcpy(&(pmb->ppar->realprop(k, 0)), &(recvbuf[os]), size);
      os += size;
    }
    for (int k = 0; k < naux; ++k) {
#pragma ivdep
      std::memcpy(&(pmb->ppar->auxprop(k, 0)), &(recvbuf[os]), size);
      os += size;
    }

    // Then eliminate the particles not belonged to the finer block
    const Real x1min = pmb->block_size.x1min, x1max = pmb->block_size.x1max;
    const Real x2min = pmb->block_size.x2min, x2max = pmb->block_size.x2max;
    const Real x3min = pmb->block_size.x3min, x3max = pmb->block_size.x3max;
    const bool active1 = pmb->ppar->active1, active2 = pmb->ppar->active2,
               active3 = pmb->ppar->active3;
    for (int k = 0; k < pmb->ppar->npar;) {
      Real x1, x2, x3;
      pmb->pcoord->CartesianToMeshCoords(pmb->ppar->xp(k), pmb->ppar->yp(k),
                                         pmb->ppar->zp(k), x1, x2, x3);
      if ((!active1 || (active1 && x1min <= x1 && x1 < x1max)) &&
          (!active2 || (active2 && x2min <= x2 && x2 < x2max)) &&
          (!active3 || (active3 && x3min <= x3 && x3 < x3max)))
        ++k;
      else
        pmb->ppar->RemoveOneParticle(k);
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::Initialize(Mesh *pm, ParameterInput *pin)
//  \brief initializes the class.

void Particles::Initialize(Mesh *pm, ParameterInput *pin) {
  if (initialized) return;

  // Add particle ID.
  ipid = AddIntProperty();
  iinit_mbid = AddIntProperty();
  // Add particle property ID.
  itid = AddIntProperty();

  // Add particle position.
  ixp = AddRealProperty();
  iyp = AddRealProperty();
  izp = AddRealProperty();

  // Add particle velocity.
  ivpx = AddRealProperty();
  ivpy = AddRealProperty();
  ivpz = AddRealProperty();

  // Add old particle position.
  ixp0 = AddAuxProperty();
  iyp0 = AddAuxProperty();
  izp0 = AddAuxProperty();

  // Add old particle velocity.
  ivpx0 = AddAuxProperty();
  ivpy0 = AddAuxProperty();
  ivpz0 = AddAuxProperty();
  iep0 = AddAuxProperty();

  // Add particle position indices.
  ixi1 = AddWorkingArray();
  ixi2 = AddWorkingArray();
  ixi3 = AddWorkingArray();

  idmax_array.NewAthenaArray(pm->nbtotal);

  // Initiate ParticleMesh class.
  ParticleMesh::Initialize(pm, pin);

  delta_f_enable = false;
  initialized = true;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::PostInitialize(Mesh *pm, ParameterInput *pin)
//  \brief preprocesses the class after problem generator and before the main loop.

void Particles::PostInitialize(Mesh *pm, ParameterInput *pin) {
  // Set particle IDs.
  SetIDMax(pm);
  ProcessNewParticles(pm);

  // Set position indices.
  for (int i = 0; i < pm->nblocal; ++i)
    pm->my_blocks(i)->ppar->SetPositionIndices();

  // Initiate ParBoundaryVariable class.
  ParBoundaryVariable::Initialize(nint, nreal, naux);
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::FindHistoryOutput(Mesh *pm, Real data_sum[], int pos)
//  \brief finds the data sums of history output from particles in my process and assign
//    them to data_sum beginning at index pos.

void Particles::FindHistoryOutput(Mesh *pm, Real data_sum[], int pos) {
  const int NSUM = NHISTORY_PAR - 1;

  // Initiate the summations.
  std::int64_t np = 0;
  std::vector<Real> sum(NSUM, 0.0);

  // Sum over each meshblock.
  Real vp1, vp2, vp3;
  for (int i = 0; i < pm->nblocal; ++i) {
    const Particles *ppar = pm->my_blocks(i)->ppar;
    np += ppar->npar;
    const Coordinates *pcoord = pm->my_blocks(i)->pcoord;
    for (int k = 0; k < ppar->npar; ++k) {
      pcoord->CartesianToMeshCoordsVector(ppar->xp(k), ppar->yp(k), ppar->zp(k),
          ppar->vpx(k), ppar->vpy(k), ppar->vpz(k), vp1, vp2, vp3);
      sum[0] += vp1;
      sum[1] += vp2;
      sum[2] += vp3;
      sum[3] += vp1 * vp1;
      sum[4] += vp2 * vp2;
      sum[5] += vp3 * vp3;
    }
  }

  // Assign the values to output variables.
  data_sum[pos++] = static_cast<Real>(np);
  for (int i = 0; i < NSUM; ++i)
    data_sum[pos++] = sum[i];
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::GetHistoryOutputNames(std::string output_names[])
//  \brief gets the names of the history output variables in history_output_names[].

void Particles::GetHistoryOutputNames(std::string output_names[]) {
  output_names[0] = "np";
  output_names[1] = "vp1";
  output_names[2] = "vp2";
  output_names[3] = "vp3";
  output_names[4] = "vp1^2";
  output_names[5] = "vp2^2";
  output_names[6] = "vp3^2";
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::GetNumberDensityOnMesh(Mesh *pm, bool include_velocity)
//  \brief finds the number density of particles on the mesh.  If include_velocity is
//    true, the velocity field is also included.

void Particles::GetNumberDensityOnMesh(Mesh *pm, bool include_velocity) {
  if (ParticleMesh::iweight < 0) return;
  // Assign particle properties to mesh and send boundary.
  ParticleMesh *ppm;
#pragma omp for private(ppm)
  for (int i = 0; i < pm->nblocal; ++i) {
    ppm = pm->my_blocks(i)->ppar->ppm;
    const Particles *ppar = pm->my_blocks(i)->ppar;
    if (ppar->npar > 0) {
      if (include_velocity && Particles::imvpx >= 0) {
        AthenaArray<Real> vp, vp1, vp2, vp3;
        vp.NewAthenaArray(3, ppar->npar);
        vp1.InitWithShallowSlice(vp, 2, 0, 1);
        vp2.InitWithShallowSlice(vp, 2, 1, 1);
        vp3.InitWithShallowSlice(vp, 2, 2, 1);
        const Coordinates *pcoord = pm->my_blocks(i)->pcoord;
        for (int k = 0; k < ppar->npar; ++k)
          pcoord->CartesianToMeshCoordsVector(
              ppar->xp(k), ppar->yp(k), ppar->zp(k), ppar->vpx(k), ppar->vpy(k),
              ppar->vpz(k), vp1(k), vp2(k), vp3(k));
        ppm->AssignParticlesToMeshAux(vp, 0, imvpx, 3);
      } else {
        ppm->AssignParticlesToMeshAux(ppar->realprop, 0, ppm->iweight, 0);
      }
    } else {
      if (include_velocity && Particles::imvpx >= 0) {
#pragma ivdep
        std::fill(&(ppar->ppm->meshaux(imvpx, 0, 0, 0)),
                  &(ppar->ppm->meshaux(imvpx, 0, 0, 0)) +
                      3 * pm->my_blocks(i)->ncells1 *
                          pm->my_blocks(i)->ncells2 * pm->my_blocks(i)->ncells3,
                  0);
      }
#pragma ivdep      
        std::fill(&(ppar->ppm->weight(0, 0, 0)),
                &(ppar->ppm->weight(0, 0, 0)) + pm->my_blocks(i)->ncells1 *
                                                    pm->my_blocks(i)->ncells2 *
                                                    pm->my_blocks(i)->ncells3,
                0);
    }
    
    ppm->pmbvar_.StartReceiving(BoundaryCommSubset::all);
    ppm->pmbvar_.SendBoundaryBuffers();
  }

#pragma omp for private(ppm)
  for (int i = 0; i < pm->nblocal; ++i) {
    Coordinates *pc = pm->my_blocks(i)->pcoord;
    ppm = pm->my_blocks(i)->ppar->ppm;
    
   // Finalize boundary communications.
    ppm->pmbvar_.ReceiveAndSetBoundariesWithWait();
    ppm->pmbvar_.ClearBoundary(BoundaryCommSubset::all);
    pm->my_blocks(i)->pbval->ApplyPhysicalBoundaries(
        pm->time, pm->dt, std::vector<BoundaryVariable *>(1, &ppm->pmbvar_));
    const int is = ppm->is, ie = ppm->ie;
    const int js = ppm->js, je = ppm->je;
    const int ks = ppm->ks, ke = ppm->ke;
    if (include_velocity) {
      // Compute the velocity field.
      for (int l = imvpx; l <= imvpz; ++l)
        for (int k = ks; k <= ke; ++k)
          for (int j = js; j <= je; ++j)
            for (int i = is; i <= ie; ++i) {
              Real w = ppm->weight(k, j, i);
              ppm->meshaux(l, k, j, i) /= (w != 0) ? w : 1;
            }
    }
    // Compute the number density.
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i)
          ppm->weight(k, j, i) /= pc->GetCellVolume(k, j, i);
  }
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::GetTotalNumber(Mesh *pm)
//  \brief returns total number of particles (from all processes).

int Particles::GetTotalNumber(Mesh *pm) {
  int npartot = 0;
  for (int i = 0; i < pm->nblocal; ++i) npartot += pm->my_blocks(i)->ppar->npar;
    
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &npartot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  return npartot;
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::EnrollUserBoundaryFunction(BoundaryFace dir, BValParFunc my_bc)
//! \brief Enroll a user-defined boundary function

void Particles::EnrollUserBoundaryFunction(BoundaryFace dir,
                                           BValParFunc my_bc) {
    std::stringstream msg;
  if (dir < 0 || dir > 5) {
    msg << "### FATAL ERROR in EnrollBoundaryCondition function" << std::endl
        << "dirName = " << dir << " not valid" << std::endl;
    ATHENA_ERROR(msg);
  }
  // Do not need to check the input file again
  /*if (mesh_bcs[dir] != BoundaryFlag::user) {
    msg << "### FATAL ERROR in EnrollUserBoundaryFunction" << std::endl
        << "The boundary condition flag must be set to the string 'user' in the "
        << " <mesh> block in the input file to use user-enrolled BCs" << std::endl;
    ATHENA_ERROR(msg);
  }*/
  BoundaryFunction_[static_cast<int>(dir)]=my_bc;
}

//--------------------------------------------------------------------------------------
//! \fn Particles::Particles(MeshBlock *pmb, ParameterInput *pin)
//  \brief constructs a Particles instance.

Particles::Particles(MeshBlock *pmb, ParameterInput *pin)
    : pmy_block(pmb),
      pmy_mesh(pmb->pmy_mesh),
      npar(0),
      nparmax(1),
      idmax(0),
      active1(pmb->pmy_mesh->mesh_size.nx1 > 1),
      active2(pmb->pmy_mesh->mesh_size.nx2 > 1),
      active3(pmb->pmy_mesh->mesh_size.nx3 > 1),
      intprop(nint, nparmax),
      realprop(nreal, nparmax),
      auxprop(naux, nparmax,
              (naux > 0) ? AthenaArray<Real>::DataStatus::allocated
                             : AthenaArray<Real>::DataStatus::empty),
      work(nwork, nparmax,
           (nwork > 0) ? AthenaArray<Real>::DataStatus::allocated
                         : AthenaArray<Real>::DataStatus::empty),
      pbvar_(pmb,&intprop,&realprop,&auxprop) {
  // Enroll ParBoundaryVariable object
  pbvar_.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&pbvar_);

  // Called before the constructor of ParticleMesh
  pmb->pbval->AdvanceCounterPhysID(ParBoundaryVariable::max_phys_id);
  // Allocate mesh auxiliaries.
  ppm = new ParticleMesh(this);

  // Shallow copy to shorthands.
  AssignShorthands();
}

//--------------------------------------------------------------------------------------
//! \fn Particles::~Particles()
//  \brief destroys a Particles instance.

Particles::~Particles() {
  // Delete integer properties.
  intprop.DeleteAthenaArray();

  // Delete real properties.
  realprop.DeleteAthenaArray();

  // Delete auxiliary properties.
  auxprop.DeleteAthenaArray();

  // Delete working arrays.
  work.DeleteAthenaArray();

  // Delete mesh auxiliaries.
  delete ppm;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ClearParBoundary(BoundaryCommSubset phase)
//  \brief resets boundary for particle transportation.

void Particles::ClearParBoundary(BoundaryCommSubset phase) {
  pbvar_.ClearBoundary(phase);
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::Integrate(int step)
//  \brief updates all particle positions and velocities from t to t + dt.

void Particles::Integrate(int stage) {
  Real t = 0, dt = 0;

  // Determine the integration cofficients.
  switch (stage) {
  case 1:
    t = pmy_mesh->time;
    dt = 0.5 * pmy_mesh->dt;
    SaveStatus();
    break;

  case 2:
    t = pmy_mesh->time + 0.5 * pmy_mesh->dt;
    dt = pmy_mesh->dt;
    break;
  }

  // Conduct one stage of the integration.
  EulerStep(t, dt, pmy_block->phydro->w);
  ReactToMeshAux(t, dt, pmy_block->phydro->w);

  // Update the position index.
  SetPositionIndices();
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::Sort() {}
//  \brief sort particles according to indices, from z to x, from lower to upper

void Particles::Sort() {
  if (npar > SIMD_WIDTH) {
    const int max_x(
        static_cast<int>(std::log2(pmy_block->block_size.nx1 + 2 * NGHOST)) +
        1),
        max_y(static_cast<int>(
                  std::log2(pmy_block->block_size.nx2 + 2 * NGHOST)) +
              1);
    int *id_list;
    id_list = new int[npar];
#pragma ivdep
    for (int i = 0; i < npar; ++i) id_list[i] = i;  // Initialize
    // Sort id_list according to particle indices
    std::sort(&id_list[0], &id_list[npar], [&](int a, int b) {
      return ((((static_cast<int>(xi3(a)) << max_y) | static_cast<int>(xi2(a)))
               << max_x) |
              static_cast<int>(xi1(a))) <
             ((((static_cast<int>(xi3(b)) << max_y) | static_cast<int>(xi2(b)))
               << max_x) |
              static_cast<int>(xi1(b)));
    });

    AthenaArray<int> int_cpy(npar);
    for (int n = 0; n < nint; ++n) {
#pragma ivdep
      std::memcpy(&int_cpy(0), &intprop(n, 0), npar * sizeof(int));
#pragma ivdep
      for (int i = 0; i < npar; ++i) intprop(n, i) = int_cpy(id_list[i]);
    }
    int_cpy.DeleteAthenaArray();

    AthenaArray<Real> real_cpy(npar);
    for (int n = 0; n < nreal; ++n) {
#pragma ivdep
      std::memcpy(&real_cpy(0), &realprop(n, 0), npar * sizeof(Real));
#pragma ivdep
      for (int i = 0; i < npar; ++i) realprop(n, i) = real_cpy(id_list[i]);
    }

    for (int n = 0; n < naux; ++n) {
#pragma ivdep
      std::memcpy(&real_cpy(0), &auxprop(n, 0), npar * sizeof(Real));
#pragma ivdep
      for (int i = 0; i < npar; ++i) auxprop(n, i) = real_cpy(id_list[i]);
    }

    for (int n = 0; n < nwork; ++n) {
#pragma ivdep
      std::memcpy(&real_cpy(0), &work(n, 0), npar * sizeof(Real));
#pragma ivdep
      for (int i = 0; i < npar; ++i) work(n, i) = real_cpy(id_list[i]);
    }

    delete[] id_list;
    real_cpy.DeleteAthenaArray();
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::LinkNeighbors()
//  \brief fetches particle neighbor information for later communication.

void Particles::LinkNeighbors() {
  // Initiate Particles boundary data.
  pbvar_.SetupPersistentMPI();
}

void Particles::InitRecvPar(BoundaryCommSubset phase) {
  pbvar_.StartReceiving(phase);
}

void Particles::SetParMeshBoundaryAttributes() {
  ppm->SetBoundaryAttributes();
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::RemoveOneParticle(int k)
//  \brief removes particle k in the block.

void Particles::RemoveOneParticle(int k) {
  if (0 <= k && k < npar && --npar != k) {
    xi1(k) = xi1(npar);
    xi2(k) = xi2(npar);
    xi3(k) = xi3(npar);
    for (int j = 0; j < nint; ++j)
      intprop(j,k) = intprop(j,npar);
    for (int j = 0; j < nreal; ++j)
      realprop(j,k) = realprop(j,npar);
    for (int j = 0; j < naux; ++j)
      auxprop(j,k) = auxprop(j,npar);
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SendParticleMesh()
//  \brief send ParticleMesh meshaux near boundaries to neighbors.

void Particles::SendParticleMesh() {
  if (backreaction) ppm->pmbvar_.SendBoundaryBuffers();
}

void Particles::ReceiveAndSetParMeshWithWait() {
  if (backreaction) ppm->pmbvar_.ReceiveAndSetBoundariesWithWait();
}

void Particles::ReceiveAndSetParWithWait() {
  pbvar_.ReceiveAndSetBoundariesWithWait();
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SendParticles()
//  \brief sends particles outside boundary to the buffers of neighboring meshblocks.
 
void Particles::SendParticles() {
  ApplyBoundaryConditions();

  // Package particles to be sent
  for (int k = 0; k < npar;) {
    const int bufid(pbvar_.BufidFromMeshIndex(xi1(k), xi2(k), xi3(k)));
    if (bufid == -1)
      ++k;  // Particle does not cross boundary
    else {
      pbvar_.LoadPar(k, bufid, false);  // Package particles
      RemoveOneParticle(k);
    }
  }

  pbvar_.SendBoundaryBuffers();
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SetPositionIndices()
//  \brief updates position indices of particles.

void Particles::SetPositionIndices() {
  GetPositionIndices(npar, xp, yp, zp, xi1, xi2, xi3);
}

//--------------------------------------------------------------------------------------
//! \fn bool Particles::ReceiveNumParAndStartRecv()
//  \brief receive the number of particles (to be received) and start receiving

bool Particles::ReceiveNumParAndStartRecv() { return pbvar_.ReceiveFluxCorrection(); }

//--------------------------------------------------------------------------------------
//! \fn bool Particles::ReceiveParticles()
//  \brief receives particles from neighboring meshblocks and returns a flag indicating
//         if all receives are completed.

bool Particles::ReceiveParticles() { return pbvar_.ReceiveBoundaryBuffers(); }

bool Particles::ReceiveParticleMesh() {
  if (!backreaction) return true;
  return ppm->pmbvar_.ReceiveBoundaryBuffers();
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SetParticleMesh(int step)
//  \brief receives ParticleMesh meshaux near boundaries from neighbors and returns a
//         flag indicating if all receives are completed.

void Particles::SetParticleMesh(int stage) {
  if (!backreaction) return;

  // Flush ParticleMesh receive buffers.
  ppm->pmbvar_.SetBoundaries();
  
  // Deposit ParticleMesh meshaux to MeshBlock.
  Hydro *phydro = pmy_block->phydro;
  Real t = 0, dt = 0;

  switch (stage) {
    case 1:
      t = pmy_mesh->time;
      dt = 0.5 * pmy_mesh->dt;
      break;

    case 2:
      t = pmy_mesh->time + 0.5 * pmy_mesh->dt;
      dt = pmy_mesh->dt;
      break;
  }

  DepositToMesh(stage, t, dt, phydro->w, phydro->u);

  return;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SetParticles
//  \brief set particle buffers to intprop, realprop and auxprop
void Particles::SetParticles() {
  const int npar_old(npar);
  pbvar_.SetBoundaries();
  ApplyLogicalConditions(npar_old);
  // Set index for those new particles from npar_old
  Real x1(0), x2(0), x3(0);
#pragma omp simd simdlen(SIMD_WIDTH) private(x1, x2, x3)
  for (int k = npar_old; k < npar; ++k) {
    // Convert to the Mesh coordinates.
    pmy_block->pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);

    // Convert to the index space.
    pmy_block->pcoord->MeshCoordsToIndices(x1, x2, x3, xi1(k), xi2(k), xi3(k));
  }
  
  if ((npar - npar_old) > 0) {
    // To determine whether to sort particles
    const int max_x(pmy_block->block_size.nx1),
        max_y(pmy_block->block_size.nx2);
    const int norm_length(npar / SIMD_WIDTH);
    int ii_max(0), begin_index(0), max_loc(-1),
        min_loc(std::numeric_limits<int>::max()), iiii(0), loc(0);
    int max_locv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
    int min_locv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
    int rel_locv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
#pragma ivdep
    std::fill(&max_locv[0], &max_locv[SIMD_WIDTH], -1);
#pragma ivdep
    std::fill(&min_locv[0], &min_locv[SIMD_WIDTH],
              std::numeric_limits<int>::max());
#pragma ivdep
    std::fill(&rel_locv[0], &rel_locv[SIMD_WIDTH], 0);

    for (int i = 0; i < norm_length; i += SIMD_WIDTH) {
      ii_max = std::min(SIMD_WIDTH, norm_length - i);
#pragma omp simd simdlen(SIMD_WIDTH) private(begin_index, max_loc, min_loc, iiii)
      for (int ii = 0; ii < ii_max; ++ii) {
        begin_index = (i + ii) * SIMD_WIDTH;
        max_loc = -1;
        min_loc = std::numeric_limits<int>::max();
#pragma loop count(SIMD_WIDTH)
        for (int iii = 0; iii < SIMD_WIDTH; ++iii) {
          iiii = begin_index + iii;
          loc = (static_cast<int>(xi3(iiii)) * max_y +
                 static_cast<int>(xi2(iiii))) *
                    max_x +
                static_cast<int>(xi1(iiii));
          max_loc = std::max(max_loc, loc);
          min_loc = std::min(min_loc, loc);
        }
        max_locv[ii] = std::max(max_locv[ii], max_loc);
        min_locv[ii] = std::min(min_locv[ii], min_loc);
        rel_locv[ii] += max_loc - min_loc;
      }
    }
    max_loc = -1;
    min_loc = std::numeric_limits<int>::max();
    int rel_loc(0);
    for (int i = 0; i < SIMD_WIDTH; ++i) {
      max_loc = std::max(max_locv[i], max_loc);
      min_loc = std::min(min_locv[i], min_loc);
      rel_loc += rel_locv[i];
    }
    // This criteria is a trial
    if (rel_loc > 300 * std::max(npar, max_loc - min_loc)) {
      /*std::cout << pmy_block->gid << ", " << rel_loc << ", " << npar << ", "
                << max_loc - min_loc
                << std::endl;*/
      Sort();
    }
  }
  
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ProcessNewParticles()
//  \brief searches for and books new particles.

void Particles::ProcessNewParticles(Mesh *pmesh) {
  // Count new particles.
  const int nbtotal = pmesh->nbtotal;
//  MeshBlock *pmb = pmesh->pblock;
//  AthenaArray<int> nnewpar;
//  nnewpar.NewAthenaArray(nbtotal);
//  while (pmb != nullptr) {
//    nnewpar(pmb->gid) = pmb->ppar->CountNewParticles();
//    pmb = pmb->next;
//  }
//#ifdef MPI_PARALLEL
//  MPI_Allreduce(MPI_IN_PLACE, &nnewpar(0), nbtotal, MPI_INT, MPI_MAX, my_comm);
//#endif
//
//  // Make the counts cumulative.
//  for (int i = 1; i < nbtotal; ++i)
//    nnewpar(i) += nnewpar(i-1);
//
//  // Set particle IDs.
//  pmb = pmesh->pblock;
//  while (pmb != nullptr) {
//    pmb->ppar->SetNewParticleID(idmax + (pmb->gid > 0 ? nnewpar(pmb->gid - 1) : 0));
//    pmb = pmb->next;
//  }
//  idmax += nnewpar(nbtotal - 1);

  for (int i = 0; i < pmesh->nblocal; ++i) {
    // Set particle IDs.
    pmesh->my_blocks(i)->ppar->SetNewParticleID();
    // Check tid.
    const int tid_max(pmesh->my_blocks(i)->ppar->GetTotalTypeNumber());
    for (int k = 0; k < pmesh->my_blocks(i)->ppar->npar; ++k)
      if (pmesh->my_blocks(i)->ppar->tid(k) >= tid_max ||
          pmesh->my_blocks(i)->ppar->tid(k) < 0) {
        std::stringstream msg;
        msg << "### FATAL ERROR in Particles::ProcessNewParticles" << std::endl
            << "In meshblock " << pmesh->my_blocks(i)->gid
            << ", one particle number " << k << ", whose tid is "
            << pmesh->my_blocks(i)->ppar->tid(k)
            << " does not belong to any type particles:"
            << " mbid is " << pmesh->my_blocks(i)->ppar->init_mbid(k)
            << " pid is " << pmesh->my_blocks(i)->ppar->pid(k)
            << " space position is (" << pmesh->my_blocks(i)->ppar->xp(k)
            << ", " << pmesh->my_blocks(i)->ppar->yp(k) << ", "
            << pmesh->my_blocks(i)->ppar->zp(k) << ") momentum is ("
            << pmesh->my_blocks(i)->ppar->vpx(k) << ", "
            << pmesh->my_blocks(i)->ppar->vpy(k) << ", "
            << pmesh->my_blocks(i)->ppar->vpz(k) << ")" << std::endl;
        ATHENA_ERROR(msg);
      }
  }
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::CountNewParticles()
//  \brief counts new particles in the block.

int Particles::CountNewParticles() const {
  int n = 0;
  for (int i = 0; i < npar; ++i)
    if (pid(i) <= 0) ++n;
  return n;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ApplyBoundaryConditions()
//  \brief applies physical boundary conditions to all particles and update mesh
//         coordinates

void Particles::ApplyBoundaryConditions() {
  // For non-periodic conditions, inlcuding user-defined
  if (pmy_block->pbval->apply_bndry_fn_[BoundaryFace::inner_x1])
    DispatchBoundaryFunctions(pmy_block, pmy_block->pcoord, pmy_mesh->time,
                              pmy_mesh->dt, BoundaryFace::inner_x1);
  if (pmy_block->pbval->apply_bndry_fn_[BoundaryFace::outer_x1])
    DispatchBoundaryFunctions(pmy_block, pmy_block->pcoord, pmy_mesh->time,
                              pmy_mesh->dt, BoundaryFace::outer_x1);
  if (active2) {
    if (pmy_block->pbval->apply_bndry_fn_[BoundaryFace::inner_x2])
      DispatchBoundaryFunctions(pmy_block, pmy_block->pcoord, pmy_mesh->time,
                                pmy_mesh->dt, BoundaryFace::inner_x2);
    if (pmy_block->pbval->apply_bndry_fn_[BoundaryFace::outer_x2])
      DispatchBoundaryFunctions(pmy_block, pmy_block->pcoord, pmy_mesh->time,
                                pmy_mesh->dt, BoundaryFace::outer_x2);
  }
  if (active3) {
    if (pmy_block->pbval->apply_bndry_fn_[BoundaryFace::inner_x3])
      DispatchBoundaryFunctions(pmy_block, pmy_block->pcoord, pmy_mesh->time,
                                pmy_mesh->dt, BoundaryFace::inner_x3);
    if (pmy_block->pbval->apply_bndry_fn_[BoundaryFace::outer_x3])
      DispatchBoundaryFunctions(pmy_block, pmy_block->pcoord, pmy_mesh->time,
                                pmy_mesh->dt, BoundaryFace::outer_x3);
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ApplyBoundaryConditions()
//  \brief applies (shear-)periodic boundary conditions, , to all particles and
//  update mesh coordinates. Including updating the inactive dimensions. Do nothing
//  for polar boundaries. Start from new particles indice_for_new

void Particles::ApplyLogicalConditions(const int &indice_for_new) {
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
    }
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void void Particles::DispatchBoundaryFunctions(MeshBlock *pmb,
//! Coordinates *pco, Real time, Real dt, BoundaryFace face)
//  \brief dealing the physical boundaries in 6 directions

void Particles::DispatchBoundaryFunctions(MeshBlock *pmb, Coordinates *pco,
                                          Real time, Real dt,
                                          BoundaryFace face) {
  if (pmy_block->pbval->block_bcs[face] ==
      BoundaryFlag::user) {  // user-enrolled BCs
    BoundaryFunction_[face](this, pmb, pco, time, dt);
  }

  std::stringstream msg;
  msg << "### FATAL ERROR in DispatchBoundaryFunctions" << std::endl
      << "face = BoundaryFace::undef passed to this function" << std::endl;

  
    switch (pmy_block->pbval->block_bcs[face]) {
    case BoundaryFlag::user:  // handled above, outside loop over
                              // BoundaryVariable objs
      break;
    case BoundaryFlag::reflect:
      switch (face) {
        case BoundaryFace::undef:
          ATHENA_ERROR(msg);
        case BoundaryFace::inner_x1:
          ReflectInnerX1(time, dt);
          break;
        case BoundaryFace::outer_x1:
          ReflectOuterX1(time, dt);
          break;
        case BoundaryFace::inner_x2:
          ReflectInnerX2(time, dt);
          break;
        case BoundaryFace::outer_x2:
          ReflectOuterX2(time, dt);
          break;
        case BoundaryFace::inner_x3:
          ReflectInnerX3(time, dt);
          break;
        case BoundaryFace::outer_x3:
          ReflectOuterX3(time, dt);
          break;
      }
      break;
    case BoundaryFlag::outflow:
      switch (face) {
        case BoundaryFace::undef:
          ATHENA_ERROR(msg);
        case BoundaryFace::inner_x1:
          OutflowInnerX1(time, dt);
          break;
        case BoundaryFace::outer_x1:
          OutflowOuterX1(time, dt);
          break;
        case BoundaryFace::inner_x2:
          OutflowInnerX2(time, dt);
          break;
        case BoundaryFace::outer_x2:
          OutflowOuterX2(time, dt);
          break;
        case BoundaryFace::inner_x3:
          OutflowInnerX3(time, dt);
          break;
        case BoundaryFace::outer_x3:
          OutflowOuterX3(time, dt);
          break;
      }
      break;
    case BoundaryFlag::polar_wedge:
      switch (face) {
        case BoundaryFace::undef:
          ATHENA_ERROR(msg);
        case BoundaryFace::inner_x2:
          PolarWedgeX2(time, dt);
          break;
        case BoundaryFace::outer_x2:
          PolarWedgeX2(time, dt);
          break;
        default:
          std::stringstream msg_polar;
          msg_polar << "### FATAL ERROR in DispatchBoundaryFunctions"
                    << std::endl
                    << "Attempting to call polar wedge boundary function on \n"
                    << "MeshBlock boundary other than inner x2 or outer x2"
                    << std::endl;
          ATHENA_ERROR(msg_polar);
      }
      break;
    default:
      std::stringstream msg_flag;
      msg_flag << "### FATAL ERROR in DispatchBoundaryFunctions" << std::endl
               << "No BoundaryPhysics function associated with provided\n"
               << "block_bcs[" << face << "] = BoundaryFlag::"
               << GetBoundaryString(pmy_block->pbval->block_bcs[face])
               << std::endl;
      ATHENA_ERROR(msg);
      break;
  }  // end switch (block_bcs[face])
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::EulerStep(Real t, Real dt, const AthenaArray<Real>& meshsrc)
//  \brief evolves the particle positions and velocities by one Euler step.

void Particles::EulerStep(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  // Update positions.
  for (int k = 0; k < npar; ++k) {
    // TODO(ccyang): This is a temporary hack.
    Real tmpx = xp(k), tmpy = yp(k), tmpz = zp(k);
    xp(k) = xp0(k) + dt * vpx(k);
    if (active2) yp(k) = yp0(k) + dt * vpy(k);
    if (active3) zp(k) = zp0(k) + dt * vpz(k);
    xp0(k) = tmpx;
    yp0(k) = tmpy;
    zp0(k) = tmpz;
  }

  // Integrate the source terms (e.g., acceleration).
  SourceTerms(t, dt, meshsrc);
  UserSourceTerms(t, dt, meshsrc);
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::GetPositionIndices(int npar,
//                                         const AthenaArray<Real>& xp,
//                                         const AthenaArray<Real>& yp,
//                                         const AthenaArray<Real>& zp,
//                                         AthenaArray<Real>& xi1,
//                                         AthenaArray<Real>& xi2,
//                                         AthenaArray<Real>& xi3)
//  \brief finds the position indices of each particle with respect to the local grid.

void Particles::GetPositionIndices(int npar,
                                   const AthenaArray<Real>& xp,
                                   const AthenaArray<Real>& yp,
                                   const AthenaArray<Real>& zp,
                                   AthenaArray<Real>& xi1,
                                   AthenaArray<Real>& xi2,
                                   AthenaArray<Real>& xi3) {
  Real x1(0), x2(0), x3(0);
#pragma omp simd simdlen(SIMD_WIDTH) private(x1, x2, x3)
  for (int k = 0; k < npar; ++k) {
    // Convert to the Mesh coordinates.
    pmy_block->pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);

    // Convert to the index space.
    pmy_block->pcoord->MeshCoordsToIndices(x1, x2, x3, xi1(k), xi2(k), xi3(k));
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SetNewParticleID(int id0)
//  \brief searches for new particles and assigns ID, beginning at id + 1.

//void Particles::SetNewParticleID(int id) {
//  for (int i = 0; i < npar; ++i)
//    if (pid(i) <= 0) pid(i) = ++id;
//}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SetNewParticleID()
//  \brief searches for new particles and assigns ID, beginning at idmax + 1.

void Particles::SetNewParticleID() {
  for (int i = 0; i < npar; ++i)
    if (pid(i) <= 0) {
      pid(i) = ++idmax;
      init_mbid(i) = this->pmy_block->gid;
    }
}

void Particles::ReflectInnerX1(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3, x10, x20, x30;
  Real vp1, vp2, vp3, vp10, vp20, vp30;
  for (int k = 0; k < npar; ++k) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (x1 < mesh_size.x1min) {
      pcoord->CartesianToMeshCoords(xp0(k), yp0(k), zp0(k), x10, x20, x30);
      pcoord->CartesianToMeshCoordsVector(xp(k), yp(k), zp(k), vpx(k), vpy(k),
                                          vpz(k), vp1, vp2, vp3);
      pcoord->CartesianToMeshCoordsVector(xp0(k), yp0(k), zp0(k), vpx0(k),
                                          vpy0(k), vpz0(k), vp10, vp20, vp30);
      x1 = 2.0 * mesh_size.x1min - x1;
      x10 = 2.0 * mesh_size.x1min - x10;
      pcoord->MeshCoordsToCartesian(x1, x2, x3, xp(k), yp(k), zp(k));
      pcoord->MeshCoordsToCartesian(x10, x20, x30, xp0(k), yp0(k), zp0(k));
      pcoord->MeshCoordsToIndices(x1, x2, x3, xi1(k), xi2(k), xi3(k));
      pcoord->MeshCoordsToCartesianVector(x1, x2, x3, -vp1, vp2, vp3, vpx(k),
                                          vpy(k), vpz(k));
      pcoord->MeshCoordsToCartesianVector(x10, x20, x30, -vp10, vp20, vp30,
                                          vpx0(k), vpy0(k), vpz0(k));
    }
  }
}

void Particles::ReflectOuterX1(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3, x10, x20, x30;
  Real vp1, vp2, vp3, vp10, vp20, vp30;
  for (int k = 0; k < npar; ++k) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (x1 > mesh_size.x1max) {
      pcoord->CartesianToMeshCoords(xp0(k), yp0(k), zp0(k), x10, x20, x30);
      pcoord->CartesianToMeshCoordsVector(xp(k), yp(k), zp(k), vpx(k), vpy(k),
                                          vpz(k), vp1, vp2, vp3);
      pcoord->CartesianToMeshCoordsVector(xp0(k), yp0(k), zp0(k), vpx0(k),
                                          vpy0(k), vpz0(k), vp10, vp20, vp30);
      x1 = 2.0 * mesh_size.x1max - x1;
      x10 = 2.0 * mesh_size.x1max - x10;
      pcoord->MeshCoordsToCartesian(x1, x2, x3, xp(k), yp(k), zp(k));
      pcoord->MeshCoordsToCartesian(x10, x20, x30, xp0(k), yp0(k), zp0(k));
      pcoord->MeshCoordsToIndices(x1, x2, x3, xi1(k), xi2(k), xi3(k));
      pcoord->MeshCoordsToCartesianVector(x1, x2, x3, -vp1, vp2, vp3, vpx(k),
                                          vpy(k), vpz(k));
      pcoord->MeshCoordsToCartesianVector(x10, x20, x30, -vp10, vp20, vp30,
                                          vpx0(k), vpy0(k), vpz0(k));
    }
  }
}

void Particles::ReflectInnerX2(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3, x10, x20, x30;
  Real vp1, vp2, vp3, vp10, vp20, vp30;
  for (int k = 0; k < npar; ++k) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (x2 < mesh_size.x2min) {
      pcoord->CartesianToMeshCoords(xp0(k), yp0(k), zp0(k), x10, x20, x30);
      pcoord->CartesianToMeshCoordsVector(xp(k), yp(k), zp(k), vpx(k), vpy(k),
                                          vpz(k), vp1, vp2, vp3);
      pcoord->CartesianToMeshCoordsVector(xp0(k), yp0(k), zp0(k), vpx0(k),
                                          vpy0(k), vpz0(k), vp10, vp20, vp30);
      x2 = 2.0 * mesh_size.x2min - x2;
      x20 = 2.0 * mesh_size.x2min - x20;
      pcoord->MeshCoordsToCartesian(x1, x2, x3, xp(k), yp(k), zp(k));
      pcoord->MeshCoordsToCartesian(x10, x20, x30, xp0(k), yp0(k), zp0(k));
      pcoord->MeshCoordsToIndices(x1, x2, x3, xi1(k), xi2(k), xi3(k));
      pcoord->MeshCoordsToCartesianVector(x1, x2, x3, vp1, -vp2, vp3, vpx(k),
                                          vpy(k), vpz(k));
      pcoord->MeshCoordsToCartesianVector(x10, x20, x30, vp10, -vp20, vp30,
                                          vpx0(k), vpy0(k), vpz0(k));
    }
  }
}

void Particles::ReflectOuterX2(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3, x10, x20, x30;
  Real vp1, vp2, vp3, vp10, vp20, vp30;
  for (int k = 0; k < npar; ++k) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (x2 > mesh_size.x2max) {
      pcoord->CartesianToMeshCoords(xp0(k), yp0(k), zp0(k), x10, x20, x30);
      pcoord->CartesianToMeshCoordsVector(xp(k), yp(k), zp(k), vpx(k), vpy(k),
                                          vpz(k), vp1, vp2, vp3);
      pcoord->CartesianToMeshCoordsVector(xp0(k), yp0(k), zp0(k), vpx0(k),
                                          vpy0(k), vpz0(k), vp10, vp20, vp30);
      x2 = 2.0 * mesh_size.x2max - x2;
      x20 = 2.0 * mesh_size.x2max - x20;
      pcoord->MeshCoordsToCartesian(x1, x2, x3, xp(k), yp(k), zp(k));
      pcoord->MeshCoordsToCartesian(x10, x20, x30, xp0(k), yp0(k), zp0(k));
      pcoord->MeshCoordsToIndices(x1, x2, x3, xi1(k), xi2(k), xi3(k));
      pcoord->MeshCoordsToCartesianVector(x1, x2, x3, vp1, -vp2, vp3, vpx(k),
                                          vpy(k), vpz(k));
      pcoord->MeshCoordsToCartesianVector(x10, x20, x30, vp10, -vp20, vp30,
                                          vpx0(k), vpy0(k), vpz0(k));
    }
  }
}

void Particles::ReflectInnerX3(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3, x10, x20, x30;
  Real vp1, vp2, vp3, vp10, vp20, vp30;
  for (int k = 0; k < npar; ++k) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (x3 < mesh_size.x3min) {
      pcoord->CartesianToMeshCoords(xp0(k), yp0(k), zp0(k), x10, x20, x30);
      pcoord->CartesianToMeshCoordsVector(xp(k), yp(k), zp(k), vpx(k), vpy(k),
                                          vpz(k), vp1, vp2, vp3);
      pcoord->CartesianToMeshCoordsVector(xp0(k), yp0(k), zp0(k), vpx0(k),
                                          vpy0(k), vpz0(k), vp10, vp20, vp30);
      x3 = 2.0 * mesh_size.x3min - x3;
      x30 = 2.0 * mesh_size.x3min - x30;
      pcoord->MeshCoordsToCartesian(x1, x2, x3, xp(k), yp(k), zp(k));
      pcoord->MeshCoordsToCartesian(x10, x20, x30, xp0(k), yp0(k), zp0(k));
      pcoord->MeshCoordsToIndices(x1, x2, x3, xi1(k), xi2(k), xi3(k));
      pcoord->MeshCoordsToCartesianVector(x1, x2, x3, vp1, vp2, -vp3, vpx(k),
                                          vpy(k), vpz(k));
      pcoord->MeshCoordsToCartesianVector(x10, x20, x30, vp10, vp20, -vp30,
                                          vpx0(k), vpy0(k), vpz0(k));
    }
  }
}

void Particles::ReflectOuterX3(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3, x10, x20, x30;
  Real vp1, vp2, vp3, vp10, vp20, vp30;
  for (int k = 0; k < npar; ++k) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (x3 > mesh_size.x3max) {
      pcoord->CartesianToMeshCoords(xp0(k), yp0(k), zp0(k), x10, x20, x30);
      pcoord->CartesianToMeshCoordsVector(xp(k), yp(k), zp(k), vpx(k), vpy(k),
                                          vpz(k), vp1, vp2, vp3);
      pcoord->CartesianToMeshCoordsVector(xp0(k), yp0(k), zp0(k), vpx0(k),
                                          vpy0(k), vpz0(k), vp10, vp20, vp30);
      x3 = 2.0 * mesh_size.x3max - x3;
      x30 = 2.0 * mesh_size.x3max - x30;
      pcoord->MeshCoordsToCartesian(x1, x2, x3, xp(k), yp(k), zp(k));
      pcoord->MeshCoordsToCartesian(x10, x20, x30, xp0(k), yp0(k), zp0(k));
      pcoord->MeshCoordsToIndices(x1, x2, x3, xi1(k), xi2(k), xi3(k));
      pcoord->MeshCoordsToCartesianVector(x1, x2, x3, vp1, vp2, -vp3, vpx(k),
                                          vpy(k), vpz(k));
      pcoord->MeshCoordsToCartesianVector(x10, x20, x30, vp10, vp20, -vp30,
                                          vpx0(k), vpy0(k), vpz0(k));
    }
  }
}

void Particles::OutflowInnerX1(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3;
  for (int k = 0; k < npar;) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (x1 < mesh_size.x1min)
      RemoveOneParticle(k);
    else
      ++k;
  }
}

void Particles::OutflowOuterX1(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3;
  for (int k = 0; k < npar;) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (x1 > mesh_size.x1max)
      RemoveOneParticle(k);
    else
      ++k;
  }
}

void Particles::OutflowInnerX2(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3;
  for (int k = 0; k < npar;) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (x2 < mesh_size.x2min)
      RemoveOneParticle(k);
    else
      ++k;
  }
}

void Particles::OutflowOuterX2(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3;
  for (int k = 0; k < npar;) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (x2 > mesh_size.x2max)
      RemoveOneParticle(k);
    else
      ++k;
  }
}

void Particles::OutflowInnerX3(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3;
  for (int k = 0; k < npar;) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (x3 < mesh_size.x3min)
      RemoveOneParticle(k);
    else
      ++k;
  }
}

void Particles::OutflowOuterX3(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3;
  for (int k = 0; k < npar;) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (x3 > mesh_size.x3max)
      RemoveOneParticle(k);
    else
      ++k;
  }
}

void Particles::PolarWedgeX2(const Real &time, const Real &dt) {
  RegionSize &mesh_size = pmy_mesh->mesh_size;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x1, x2, x3, x10, x20, x30;
  Real vp1, vp2, vp3, vp10, vp20, vp30;
  // Asssuming uniform mesh
  Real phi_pole_low(mesh_size.x3min - NGHOST * pcoord->dx3v(pmy_block->ks) +
                    PI),
      phi_pole_up(mesh_size.x3max + NGHOST * pcoord->dx3v(pmy_block->ke) + PI);
  if (phi_pole_low > TWO_PI) phi_pole_low -= TWO_PI;
  if (phi_pole_up > TWO_PI) phi_pole_up -= TWO_PI;
  // the ghost zone crossing polar locating at the axis (or not)
  std::function<bool(Real)> func_cross_pole;
  if (phi_pole_low > phi_pole_up)
    func_cross_pole = [&](const Real &phi) {
      return (phi > phi_pole_low) || (phi < phi_pole_up);
    };
  else
    func_cross_pole = [&](const Real &phi) {
      return (phi > phi_pole_low) && (phi < phi_pole_up);
    };

  for (int k = 0; k < npar; ++k) {
    pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);
    if (func_cross_pole(x3)) {
      pcoord->CartesianToMeshCoords(xp0(k), yp0(k), zp0(k), x10, x20, x30);
      pcoord->CartesianToMeshCoordsVector(xp(k), yp(k), zp(k), vpx(k), vpy(k),
                                          vpz(k), vp1, vp2, vp3);
      pcoord->CartesianToMeshCoordsVector(xp0(k), yp0(k), zp0(k), vpx0(k),
                                          vpy0(k), vpz0(k), vp10, vp20, vp30);
      if (x3 > PI) {
        x3 -= PI;
        x30 -= PI;
      } else {
        x3 += PI;
        x30 += PI;
      }
      pcoord->MeshCoordsToCartesian(x1, x2, x3, xp(k), yp(k), zp(k));
      pcoord->MeshCoordsToCartesian(x10, x20, x30, xp0(k), yp0(k), zp0(k));
      pcoord->MeshCoordsToIndices(x1, x2, x3, xi1(k), xi2(k), xi3(k));
      pcoord->MeshCoordsToCartesianVector(x1, x2, x3, vp1, vp2, vp3, vpx(k),
                                          vpy(k), vpz(k));
      pcoord->MeshCoordsToCartesianVector(x10, x20, x30, vp10, vp20, -vp30,
                                          vpx0(k), vpy0(k), vpz0(k));
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SaveStatus()
//  \brief saves the current positions and velocities for later use.

void Particles::SaveStatus() {
#pragma ivdep
  std::memcpy(&xp0(0), &xp(0), npar * sizeof(Real));
#pragma ivdep
  std::memcpy(&yp0(0), &yp(0), npar * sizeof(Real));
#pragma ivdep
  std::memcpy(&zp0(0), &zp(0), npar * sizeof(Real));
#pragma ivdep
  std::memcpy(&vpx0(0), &vpx(0), npar * sizeof(Real));
#pragma ivdep
  std::memcpy(&vpy0(0), &vpy(0), npar * sizeof(Real));
#pragma ivdep
  std::memcpy(&vpz0(0), &vpz(0), npar * sizeof(Real));
#pragma ivdep
  std::fill(&ep0(0), &ep0(npar), 0.0);
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddIntProperty()
//  \brief adds one integer property to the particles and returns the index.

int Particles::AddIntProperty() {
  return nint++;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddRealProperty()
//  \brief adds one real property to the particles and returns the index.

int Particles::AddRealProperty() {
  return nreal++;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddAuxProperty()
//  \brief adds one auxiliary property to the particles and returns the index.

int Particles::AddAuxProperty() {
  return naux++;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddWorkingArray()
//  \brief adds one working array to the particles and returns the index.

int Particles::AddWorkingArray() {
  return nwork++;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::AssignShorthands()
//  \brief assigns shorthands by shallow copying slices of the data.

void Particles::AssignShorthands() {
  pid.InitWithShallowSlice(intprop, 2, ipid, 1);
  init_mbid.InitWithShallowSlice(intprop, 2, iinit_mbid, 1);

  xp.InitWithShallowSlice(realprop, 2, ixp, 1);
  yp.InitWithShallowSlice(realprop, 2, iyp, 1);
  zp.InitWithShallowSlice(realprop, 2, izp, 1);
  vpx.InitWithShallowSlice(realprop, 2, ivpx, 1);
  vpy.InitWithShallowSlice(realprop, 2, ivpy, 1);
  vpz.InitWithShallowSlice(realprop, 2, ivpz, 1);
  if (delta_f_enable) {
    inv_f0.InitWithShallowSlice(realprop, 2, if0, 1);
    delta_f_weight.InitWithShallowSlice(work, 2, idelta_f_weight, 1);
  }
  tid.InitWithShallowSlice(intprop, 2, itid, 1);

  xp0.InitWithShallowSlice(auxprop, 2, ixp0, 1);
  yp0.InitWithShallowSlice(auxprop, 2, iyp0, 1);
  zp0.InitWithShallowSlice(auxprop, 2, izp0, 1);
  vpx0.InitWithShallowSlice(auxprop, 2, ivpx0, 1);
  vpy0.InitWithShallowSlice(auxprop, 2, ivpy0, 1);
  vpz0.InitWithShallowSlice(auxprop, 2, ivpz0, 1);
  ep0.InitWithShallowSlice(auxprop, 2, iep0, 1);

  xi1.InitWithShallowSlice(work, 2, ixi1, 1);
  xi2.InitWithShallowSlice(work, 2, ixi2, 1);
  xi3.InitWithShallowSlice(work, 2, ixi3, 1);
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::UpdateCapacity(int new_nparmax)
//  \brief changes the capacity of particle arrays while preserving existing data.

void Particles::UpdateCapacity(int new_nparmax) {
  // Increase size of property arrays
  nparmax = new_nparmax;
  intprop.ResizeLastDimension(nparmax);
  realprop.ResizeLastDimension(nparmax);
  if (naux > 0) auxprop.ResizeLastDimension(nparmax);
  if (nwork > 0) work.ResizeLastDimension(nparmax);

  // Reassign the shorthands.
  AssignShorthands();
}

//--------------------------------------------------------------------------------------
//! \fn Real Particles::NewBlockTimeStep();
//  \brief returns the time step required by particles in the block.

Real Particles::NewBlockTimeStep() const {
  Coordinates *pc = pmy_block->pcoord;

  // Find the maximum coordinate speed.
  Real dt_inv2_max = 0.0;
  for (int k = 0; k < npar; ++k) {
    Real dt_inv2 = 0.0, vpx1, vpx2, vpx3;
    pc->CartesianToMeshCoordsVector(xp(k), yp(k), zp(k), vpx(k), vpy(k), vpz(k),
                                    vpx1, vpx2, vpx3);
    dt_inv2 += active1 ? std::pow(vpx1 / pc->dx1f(static_cast<int>(xi1(k))), 2) : 0;
    dt_inv2 += active2 ? std::pow(vpx2 / pc->dx2f(static_cast<int>(xi2(k))), 2) : 0;
    dt_inv2 += active3 ? std::pow(vpx3 / pc->dx3f(static_cast<int>(xi3(k))), 2) : 0;
    dt_inv2_max = std::max(dt_inv2_max, dt_inv2);
  }

  // Return the time step constrained by the coordinate speed.
  return dt_inv2_max > 0.0 ? cfl_par / std::sqrt(dt_inv2_max)
                           : std::numeric_limits<Real>::max();
}

//--------------------------------------------------------------------------------------
//! \fn Real Particles::Statistics(std::function<Real(Real, Real)> const &func) const
//  \brief count the statistical value, using the function func to calculate

Real Particles::Statistics(std::function<Real(Real, Real)> const &func) const {
  return Real();
}

bool Particles::DeltafEnable() { return delta_f_enable; }

bool Particles::BackReactionEnable() { return backreaction; }

bool Particles::AssignWeightEnable() { return (ParticleMesh::iweight > -1); }

int Particles::Index_X_Vel() { return Particles::imvpx; }

int Particles::Index_Y_Vel() { return Particles::imvpy; }

int Particles::Index_Z_Vel() { return Particles::imvpz; }

//--------------------------------------------------------------------------------------
//! \fn Particles::GetSizeInBytes()
//  \brief returns the data size in bytes in the meshblock.

std::size_t Particles::GetSizeInBytes() {
  std::size_t size = sizeof(npar) + sizeof(idmax);
  if (npar > 0) size += npar * (nint * sizeof(int) + nreal * sizeof(Real));
  return size;
}

//--------------------------------------------------------------------------------------
//! \fn Particles::UnpackParticlesForRestart()
//  \brief reads the particle data from the restart file.

void Particles::UnpackParticlesForRestart(char *mbdata, std::size_t &os) {
  // Read number of particles.
  std::memcpy(&npar, &(mbdata[os]), sizeof(npar));
  os += sizeof(npar);
  std::memcpy(&idmax, &(mbdata[os]), sizeof(idmax));
  os += sizeof(idmax);
  if (nparmax < npar)
    UpdateCapacity(npar);

  if (npar > 0) {
    // Read integer properties.
    std::size_t size = npar * sizeof(int);
    for (int k = 0; k < nint; ++k) {
      std::memcpy(&(intprop(k,0)), &(mbdata[os]), size);
      os += size;
    }

    // Read real properties.
    size = npar * sizeof(Real);
    for (int k = 0; k < nreal; ++k) {
      std::memcpy(&(realprop(k,0)), &(mbdata[os]), size);
      os += size;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn Particles::PackParticlesForRestart()
//  \brief pack the particle data for restart dump.

void Particles::PackParticlesForRestart(char *&pdata) {
  // Write number of particles.
  std::memcpy(pdata, &npar, sizeof(npar));
  pdata += sizeof(npar);
  std::memcpy(pdata, &idmax, sizeof(idmax));
  pdata += sizeof(idmax);

  if (npar > 0) {
    // Write integer properties.
    std::size_t size = npar * sizeof(int);
    for (int k = 0; k < nint; ++k) {
      std::memcpy(pdata, &(intprop(k,0)), size);
      pdata += size;
    }
    // Write real properties.
    size = npar * sizeof(Real);
    for (int k = 0; k < nreal; ++k) {
      std::memcpy(pdata, &(realprop(k,0)), size);
      pdata += size;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn Particles::FormattedTableOutput()
//  \brief outputs the particle data in tabulated format.

void Particles::FormattedTableOutput(Mesh *pm, OutputParameters op) {
  Particles *ppar;
  std::stringstream fname, msg;
  std::ofstream os;

  // Loop over MeshBlocks
  for (int i = 0; i < pm->nblocal; ++i) {
    ppar = pm->my_blocks(i)->ppar;

    // Create the filename.
    fname << op.file_basename << ".block" << pm->my_blocks(i)->gid << '.'
          << op.file_id << '.' << std::setw(5) << std::right
          << std::setfill('0') << op.file_number << '.' << "par.tab";

    // Open the file for write.
    os.open(fname.str().data());
    if (!os.is_open()) {
      msg << "### FATAL ERROR in function [Particles::FormattedTableOutput]"
          << std::endl << "Output file '" << fname.str() << "' could not be opened"
          << std::endl;
      ATHENA_ERROR(msg);
    }

    // Write the time.
    os << std::setprecision(18);
    os << "# Athena++ particle data at time = " << pm->time << std::endl;
    os << "born_meshblock"
       << "  "
       << "particle_id"
       << "  "
       << "x"
       << "  "
       << "y"
       << "  "
       << "z"
       << "  "
       << "vx"
       << "  "
       << "vy"
       << "  "
       << "vz" 
       << "  "
       << "Bx"
       << "  "
       << "By"
       << "  "
       << "Bz"
       << std::endl;

    // Write the particle data in the meshblock.
    for (int k = 0; k < ppar->npar; ++k)
      if ((ppar->pid(k) % 32000) == 0) {
        int pos_x1 = static_cast<int>(ppar->xi1(k));
        int pos_x2 = static_cast<int>(ppar->xi2(k));
        int pos_x3 = static_cast<int>(ppar->xi3(k));

        Real B1 = pm->my_blocks(i)->pfield->bcc(IB1, pos_x3, pos_x2, pos_x1);
        Real B2 = pm->my_blocks(i)->pfield->bcc(IB2, pos_x3, pos_x2, pos_x1);
        Real B3 = pm->my_blocks(i)->pfield->bcc(IB3, pos_x3, pos_x2, pos_x1);
        
        os << ppar->init_mbid(k) << "  " << ppar->pid(k) << "  "
          << ppar->xp(k) << "  " << ppar->yp(k) << "  " << ppar->zp(k) << "  "
          << ppar->vpx(k) << "  " << ppar->vpy(k) << "  " << ppar->vpz(k) << "  " 
          << B1 << "  " << B2 << "  " << B3 << std::endl;
      }

    // Close the file and get the next meshblock.
    os.close();
    fname.str("");
  }
}

//--------------------------------------------------------------------------------------
//! \fn Particles::BinaryOutput(Mesh *pm, OutputParameters op)
//  \brief outputs the particle data in a binary format.

void Particles::BinaryOutput(Mesh *pm, OutputParameters op) {
  Particles *ppar;
  std::stringstream fname, msg;
  std::ofstream os;
  BufferBox buffer_box = {};
  buffer_box[ImXMin] = pm->mesh_size.x1min;
  buffer_box[ImXMax] = pm->mesh_size.x1max;
  buffer_box[ImYMin] = pm->mesh_size.x2min;
  buffer_box[ImYMax] = pm->mesh_size.x2max;
  buffer_box[ImZMin] = pm->mesh_size.x3min;
  buffer_box[ImZMax] = pm->mesh_size.x3max;
  BufferTime buffer_time = {};
  buffer_time[ITime] = pm->time;
  buffer_time[Idt] = pm->dt;

  // Loop over MeshBlocks
  for (int i = 0; i < pm->nblocal; ++i) {
    ppar = pm->my_blocks(i)->ppar;

    // Create the filename.
    fname << op.file_basename << ".block" << pm->my_blocks(i)->gid << '.'
          << op.file_id << '.' << std::setw(5) << std::right
          << std::setfill('0') << op.file_number << '.' << "par.bin";

    // Open the file for write.
    os.open(fname.str().data(), std::ios::binary | std::ios::out);
    if (!os.is_open()) {
      msg << "### FATAL ERROR in function [Particles::BinaryOutput]"
          << std::endl
          << "Output file '" << fname.str() << "' could not be opened"
          << std::endl;
      ATHENA_ERROR(msg);
    }

    // Write the grid and domain boundary
    buffer_box[ImbXMin] = pm->my_blocks(i)->block_size.x1min;
    buffer_box[ImbXMax] = pm->my_blocks(i)->block_size.x1max;
    buffer_box[ImbYMin] = pm->my_blocks(i)->block_size.x2min;
    buffer_box[ImbYMax] = pm->my_blocks(i)->block_size.x2max;
    buffer_box[ImbZMin] = pm->my_blocks(i)->block_size.x3min;
    buffer_box[ImbZMax] = pm->my_blocks(i)->block_size.x3max;
    os.write(reinterpret_cast<char *>(buffer_box), sizeof(buffer_box));

    //// Write particle property information
    //// TODO(sxc18): implement for multiple kinds of particles
    //std::vector<Real> par_types = ppar->GetTypes();
    //int npar_types(par_types.size());
    //BufferType buffer_type = {};
    //os.write(reinterpret_cast<char *>(&npar_types), sizeof(npar_types));
    //for (std::size_t i = 0; i < npar_types; ++i) buffer_type[i] = par_types[i];
    //os.write(reinterpret_cast<char *>(buffer_type), sizeof(float) * npar_types);
    
    // Write the time.
    os.write(reinterpret_cast<char *>(buffer_time), sizeof(buffer_time));

    // Write particle number
    long npar_total(ppar->npar);
    os.write(reinterpret_cast<char *>(&npar_total), sizeof(npar_total));

    // Write the particle data in the meshblock.
    BufferPar buffer_par;
    for (int k = 0; k < ppar->npar; ++k) {
      buffer_par.xp = ppar->xp(k);
      buffer_par.yp = ppar->yp(k);
      buffer_par.zp = ppar->zp(k);
      buffer_par.vpx = ppar->vpx(k);
      buffer_par.vpy = ppar->vpy(k);
      buffer_par.vpz = ppar->vpz(k);
      // TODO(sxc18): get particle density at the position (xp, yp, zp)
      buffer_par.dpar = 1.0;
      buffer_par.pid = ppar->pid(k);
      buffer_par.init_mbid = ppar->init_mbid(k);
      // TODO(sxc18): implement for multiple kinds of particles
      buffer_par.property = ppar->GetTypes(k);
      os.write(reinterpret_cast<char *>(&buffer_par), size_BufferPar);
    }

    // Close the file and get the next meshblock.
    os.close();
    fname.str("");
  }
}