//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartesian.cpp
//! \brief implements functions for Cartesian (x-y-z) coordinates in a derived class of
//! the Coordinates abstract base class.

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "coordinates.hpp"

//----------------------------------------------------------------------------------------
//! Cartesian coordinates constructor

Cartesian::Cartesian(MeshBlock *pmb, ParameterInput *pin, bool flag)
    : Coordinates(pmb, pin, flag) {
  // initialize volume-averaged coordinates and spacing
  // x1-direction: x1v = dx/2
  for (int i=il-ng; i<=iu+ng; ++i) {
    x1v(i) = 0.5*(x1f(i+1) + x1f(i));
  }
  for (int i=il-ng; i<=iu+ng-1; ++i) {
    if (pmb->block_size.x1rat != 1.0) {
      dx1v(i) = x1v(i+1) - x1v(i);
    } else {
      // dx1v = dx1f constant for uniform mesh; may disagree with x1v(i+1) - x1v(i)
      dx1v(i) = dx1f(i);
    }
  }

  // x2-direction: x2v = dy/2
  if (pmb->block_size.nx2 == 1) {
    x2v(jl) = 0.5*(x2f(jl+1) + x2f(jl));
    dx2v(jl) = dx2f(jl);
  } else {
    for (int j=jl-ng; j<=ju+ng; ++j) {
      x2v(j) = 0.5*(x2f(j+1) + x2f(j));
    }
    for (int j=jl-ng; j<=ju+ng-1; ++j) {
      if (pmb->block_size.x2rat != 1.0) {
        dx2v(j) = x2v(j+1) - x2v(j);
      } else {
        // dx2v = dx2f constant for uniform mesh; may disagree with x2v(j+1) - x2v(j)
        dx2v(j) = dx2f(j);
      }
    }
  }

  // x3-direction: x3v = dz/2
  if (pmb->block_size.nx3 == 1) {
    x3v(kl) = 0.5*(x3f(kl+1) + x3f(kl));
    dx3v(kl) = dx3f(kl);
  } else {
    for (int k=kl-ng; k<=ku+ng; ++k) {
      x3v(k) = 0.5*(x3f(k+1) + x3f(k));
    }
    for (int k=kl-ng; k<=ku+ng-1; ++k) {
      if (pmb->block_size.x3rat != 1.0) {
        dx3v(k) = x3v(k+1) - x3v(k);
      } else {
        // dxkv = dx3f constant for uniform mesh; may disagree with x3v(k+1) - x3v(k)
        dx3v(k) = dx3f(k);
      }
    }
  }
  // initialize geometry coefficients
  // x1-direction
  for (int i=il-ng; i<=iu+ng; ++i) {
    h2v(i) = 1.0;
    h2f(i) = 1.0;
    h31v(i) = 1.0;
    h31f(i) = 1.0;
    dh2vd1(i) = 0.0;
    dh2fd1(i) = 0.0;
    dh31vd1(i) = 0.0;
    dh31fd1(i) = 0.0;
  }

  // x2-direction
  if (pmb->block_size.nx2 == 1) {
    h32v(jl) = 1.0;
    h32f(jl) = 1.0;
    dh32vd2(jl) = 0.0;
    dh32fd2(jl) = 0.0;
  } else {
    for (int j=jl-ng; j<=ju+ng; ++j) {
      h32v(j) = 1.0;
      h32f(j) = 1.0;
      dh32vd2(j) = 0.0;
      dh32fd2(j) = 0.0;
    }
  }

  // initialize area-averaged coordinates used with MHD AMR
  if ((pmb->pmy_mesh->multilevel) && MAGNETIC_FIELDS_ENABLED) {
    for (int i=il-ng; i<=iu+ng; ++i) {
      x1s2(i) = x1s3(i) = x1v(i);
    }
    if (pmb->block_size.nx2 == 1) {
      x2s1(jl) = x2s3(jl) = x2v(jl);
    } else {
      for (int j=jl-ng; j<=ju+ng; ++j) {
        x2s1(j) = x2s3(j) = x2v(j);
      }
    }
    if (pmb->block_size.nx3 == 1) {
      x3s1(kl) = x3s2(kl) = x3v(kl);
    } else {
      for (int k=kl-ng; k<=ku+ng; ++k) {
        x3s1(k) = x3s2(k) = x3v(k);
      }
    }
  }
}

// functions

//--------------------------------------------------------------------------------------
//! \fn void Cartesian::CartesianToMeshCoords(
//          Real x, Real y, Real z, Real& x1, Real& x2, Real& x3) const
//  \brief returns in (x1, x2, x3) the coordinates used by the mesh from
//  Cartesian coordinates (x, y, z).

inline void Cartesian::CartesianToMeshCoords(const Real& x, const Real& y,
                                             const Real& z, Real& x1, Real& x2,
                                             Real& x3) const {
  x1 = x;
  x2 = y;
  x3 = z;
}

//--------------------------------------------------------------------------------------
//! \fn void Cartesian::MeshCoordsToCartesian(
//          Real x1, Real x2, Real x3, Real& x, Real& y, Real& z) const
//  \brief returns in Cartesian coordinates (x, y, z) from (x1, x2, x3) the
//  coordinates used by the mesh.

inline void Cartesian::MeshCoordsToCartesian(const Real& x1, const Real& x2,
                                             const Real& x3, Real& x, Real& y,
                                             Real& z) const {
  x = x1;
  y = x2;
  z = x3;
}

//--------------------------------------------------------------------------------------
//! \fn void Coordinates::CartesianToMeshCoordsVector(
//               Real x, Real y, Real z, Real vx, Real vy, Real vz,
//               Real& vx1, Real& vx2, Real& vx3)
//  \brief returns in (vx1, vx2, vx3) the components of a vector in Mesh
//  coordinates when the vector is (vx, vy, vz) at (x, y, z) in Cartesian coordinates.

inline void Cartesian::CartesianToMeshCoordsVector(
    const Real& x, const Real& y, const Real& z, const Real& vx, const Real& vy,
    const Real& vz, Real& vx1, Real& vx2, Real& vx3) const {
  vx1 = vx;
  vx2 = vy;
  vx3 = vz;
}

//--------------------------------------------------------------------------------------
//! \fn void Coordinates::MeshCoordsToCartesianVector(
//               Real x1, Real x2, Real x3, Real vx1, Real vx2, Real vx3,
//               Real& vx, Real& vy, Real& vz)
//  \brief returns in (vx, vy, vz) the components of a vector in Cartesian
//  coordinates when the vector is (vx1, vy1, vz1) at (x1, x2, x3) in Mesh coordinates.

inline void Cartesian::MeshCoordsToCartesianVector(
    const Real& x1, const Real& x2, const Real& x3, const Real& vx1,
    const Real& vx2, const Real& vx3, Real& vx, Real& vy, Real& vz) const {
  vx = vx1;
  vy = vx2;
  vz = vx3;
}
