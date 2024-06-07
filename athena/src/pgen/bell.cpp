//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bell.cpp
//! \brief Bell instability with CPAW for 1D/2D/3D problems
//!
//! In 1D, the problem is setup along one of the three coordinate axes (specified by
//! setting [ang_2,ang_3] = 0.0 or PI/2 in the input file).  In 2D/3D this routine
//! automatically sets the wavevector along the domain diagonal.
//! 
//! When charged particle module is enabled, this problem can be used fot the
//! Bell instability.
//!
//! REFERENCE: G. Toth,  "The div(B)=0 constraint in shock capturing MHD codes", JCP,
//!   161, 605 (2000)
//! A. R. Bell, "Turbulent ampliÞcation of magnetic Þeld and diffusive shock
//! acceleration of cosmic rays", MNRAS, 353, 550 (2004)

// C headers

// C++ headers
#include <algorithm>
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../particles/particles.hpp"

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

namespace {
// Parameters which define initial solution -- made global so that they can be shared
// with functions A1,2,3 which compute vector potentials
Real den, pres, gm1, b_par, b_perp, v_perp, v_par;
Real ang_2, ang_3; // Rotation angles about the y and z' axis
Real fac, sin_a2, cos_a2, sin_a3, cos_a3;
Real lambda, k_par; // Wavelength, 2*PI/wavelength
Real dx1_reciprocal, dx2_reciprocal,
    dx3_reciprocal;     // Number density of charged particles
Real vpx0, vpy0, vpz0;  // Momentum of each charged particles
Real theta;             // Phase correction in Bell instability

// functions to compute vector potential to initialize the solution
Real A1(const Real x1, const Real x2, const Real x3);
Real A2(const Real x1, const Real x2, const Real x3);
Real A3(const Real x1, const Real x2, const Real x3);
} // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Initialize magnetic field parameters
  // For wavevector along coordinate axes, set desired values of ang_2/ang_3.
  //    For example, for 1D problem use ang_2 = ang_3 = 0.0
  //    For wavevector along grid diagonal, do not input values for ang_2/ang_3.
  // Code below will automatically calculate these imposing periodicity and exactly one
  // wavelength along each grid direction
  b_par = pin->GetReal("problem","b_par");
  b_perp = pin->GetReal("problem","b_perp");
  v_par = pin->GetReal("problem","v_par");
  ang_2 = pin->GetOrAddReal("problem","ang_2",-999.9);
  ang_3 = pin->GetOrAddReal("problem","ang_3",-999.9);
  Real dir = pin->GetOrAddReal("problem","dir",1); // right(1)/left(2) polarization
  if (NON_BAROTROPIC_EOS) {
    Real gam   = pin->GetReal("hydro","gamma");
    gm1 = (gam - 1.0);
  }
  pres = pin->GetReal("problem","pres");
  den = 1.0;

  Real x1size = mesh_size.x1max - mesh_size.x1min;
  Real x2size = mesh_size.x2max - mesh_size.x2min;
  Real x3size = mesh_size.x3max - mesh_size.x3min;

  // User should never input -999.9 in angles
  if (ang_3 == -999.9) ang_3 = std::atan(x1size/x2size);
  sin_a3 = std::sin(ang_3);
  cos_a3 = std::cos(ang_3);

  if (ang_2 == -999.9) ang_2 = std::atan(0.5*(x1size*cos_a3 + x2size*sin_a3)/x3size);
  sin_a2 = std::sin(ang_2);
  cos_a2 = std::cos(ang_2);

  Real x1 = x1size*cos_a2*cos_a3;
  Real x2 = x2size*cos_a2*sin_a3;
  Real x3 = x3size*sin_a2;

  // For lambda choose the smaller of the 3
  lambda = x1;
  if (mesh_size.nx2 > 1 && ang_3 != 0.0) lambda = std::min(lambda,x2);
  if (mesh_size.nx3 > 1 && ang_2 != 0.0) lambda = std::min(lambda,x3);

  // Initialize k_parallel
  k_par = 2.0*(PI)/lambda;
  v_perp = b_perp/std::sqrt(den);

  if (dir == 1) // right polarization
    fac = 1.0;
  else          // left polarization
    fac = -1.0;

  theta = 0.0;

  // Initialize charged particles' properties
  if (PARTICLES == CHARGED_PAR) {
    // Total number of particles = npx1 * npx2 * npx3
    int npx1((mesh_size.nx1 > 1)
                 ? pin->GetOrAddInteger("problem", "npx1", mesh_size.nx1)
                 : 1),
        npx2((mesh_size.nx2 > 1)
                 ? pin->GetOrAddInteger("problem", "npx2", mesh_size.nx2)
                 : 1),
        npx3((mesh_size.nx3 > 1)
                 ? pin->GetOrAddInteger("problem", "npx3", mesh_size.nx3)
                 : 1);
    dx1_reciprocal = static_cast<Real>(npx1) / x1size;
    dx2_reciprocal = static_cast<Real>(npx2) / x2size;
    dx3_reciprocal = static_cast<Real>(npx3) / x3size;

    Real epsi(pin->GetOrAddReal("problem", "epsilon", 0.01));
    if ((epsi >= 1.0) || (epsi <= 0.0)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "[cpaw]: epsilon must be in the range of (0,1)!" << std::endl;
      ATHENA_ERROR(msg);
    }
    if (pin->GetOrAddBoolean("particles", "backreaction", false))
      theta = std::asin(epsi) + PI / 2.0;
    Real v_A(std::sqrt(1 / den) * b_par);
    Real u0(v_A / epsi);

    // CR current density
    Real j_CR(2.0 * k_par * b_par);  // CR current density that gives lambda as
                                     // the most unstable wavelength

    // particle related parameters
    Real cL(1000.0 * u0);  // reset speed of light, which is arbitrary
    pin->SetReal("particles", "speed_of_light", cL);
    Real vp_par((u0)*cL / sqrt(SQR(cL) - SQR(u0)));  // 4-velocity amplitude

    // momentum of charged particles in parallel direction
    vpx0 = vp_par * cos_a2 * cos_a3;
    vpy0 = vp_par * cos_a2 * sin_a3;
    vpz0 = vp_par * sin_a2;

    Real nCR(j_CR / u0);
    Real qomc(1.0e-3 * k_par * v_A /
              b_par);  // sufficiently small not to deflect the CRs
    pin->SetReal("particles", "charge_over_mass_over_c", qomc);

    Real mCR(nCR / qomc / npx1 / npx2 / npx3 * x1size * x2size *
             x3size);  // equivalent particle mass for momentum feedback
    pin->SetReal("particles", "mass", mCR);
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief circularly polarized Alfven wave problem generator for 1D/2D/3D problems.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  AthenaArray<Real> a1, a2, a3;
  // nxN != ncellsN, in general. Allocate to extend through ghost zones, regardless # dim
  int nx1 = block_size.nx1 + 2*NGHOST;
  int nx2 = block_size.nx2 + 2*NGHOST;
  int nx3 = block_size.nx3 + 2*NGHOST;
  a1.NewAthenaArray(nx3, nx2, nx1);
  a2.NewAthenaArray(nx3, nx2, nx1);
  a3.NewAthenaArray(nx3, nx2, nx1);

  int level = loc.level;
  // Initialize components of the vector potential
  if (block_size.nx3 > 1) {
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie+1; i++) {
          if ((pbval->nblevel[1][0][1]>level && j==js)
              || (pbval->nblevel[1][2][1]>level && j==je+1)
              || (pbval->nblevel[0][1][1]>level && k==ks)
              || (pbval->nblevel[2][1][1]>level && k==ke+1)
              || (pbval->nblevel[0][0][1]>level && j==js   && k==ks)
              || (pbval->nblevel[0][2][1]>level && j==je+1 && k==ks)
              || (pbval->nblevel[2][0][1]>level && j==js   && k==ke+1)
              || (pbval->nblevel[2][2][1]>level && j==je+1 && k==ke+1)) {
            Real x1l = pcoord->x1f(i)+0.25*pcoord->dx1f(i);
            Real x1r = pcoord->x1f(i)+0.75*pcoord->dx1f(i);
            a1(k,j,i) = 0.5*(A1(x1l, pcoord->x2f(j), pcoord->x3f(k)) +
                             A1(x1r, pcoord->x2f(j), pcoord->x3f(k)));
          } else {
            a1(k,j,i) = A1(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k));
          }

          if ((pbval->nblevel[1][1][0]>level && i==is)
              || (pbval->nblevel[1][1][2]>level && i==ie+1)
              || (pbval->nblevel[0][1][1]>level && k==ks)
              || (pbval->nblevel[2][1][1]>level && k==ke+1)
              || (pbval->nblevel[0][1][0]>level && i==is   && k==ks)
              || (pbval->nblevel[0][1][2]>level && i==ie+1 && k==ks)
              || (pbval->nblevel[2][1][0]>level && i==is   && k==ke+1)
              || (pbval->nblevel[2][1][2]>level && i==ie+1 && k==ke+1)) {
            Real x2l = pcoord->x2f(j)+0.25*pcoord->dx2f(j);
            Real x2r = pcoord->x2f(j)+0.75*pcoord->dx2f(j);
            a2(k,j,i) = 0.5*(A2(pcoord->x1f(i), x2l, pcoord->x3f(k)) +
                             A2(pcoord->x1f(i), x2r, pcoord->x3f(k)));
          } else {
            a2(k,j,i) = A2(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k));
          }

          if ((pbval->nblevel[1][1][0]>level && i==is)
              || (pbval->nblevel[1][1][2]>level && i==ie+1)
              || (pbval->nblevel[1][0][1]>level && j==js)
              || (pbval->nblevel[1][2][1]>level && j==je+1)
              || (pbval->nblevel[1][0][0]>level && i==is   && j==js)
              || (pbval->nblevel[1][0][2]>level && i==ie+1 && j==js)
              || (pbval->nblevel[1][2][0]>level && i==is   && j==je+1)
              || (pbval->nblevel[1][2][2]>level && i==ie+1 && j==je+1)) {
            Real x3l = pcoord->x3f(k)+0.25*pcoord->dx3f(k);
            Real x3r = pcoord->x3f(k)+0.75*pcoord->dx3f(k);
            a3(k,j,i) = 0.5*(A3(pcoord->x1f(i), pcoord->x2f(j), x3l) +
                             A3(pcoord->x1f(i), pcoord->x2f(j), x3r));
          } else {
            a3(k,j,i) = A3(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k));
          }
        }
      }
    }
  } else {
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie+1; i++) {
          if (i != ie+1)
            a1(k,j,i) = A1(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k));
          if (j != je+1)
            a2(k,j,i) = A2(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k));
          if (k != ke+1)
            a3(k,j,i) = A3(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k));
        }
      }
    }
  }

  // Initialize interface fields
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie+1; i++) {
        pfield->b.x1f(k,j,i) = (a3(k  ,j+1,i) - a3(k,j,i))/pcoord->dx2f(j) -
                               (a2(k+1,j  ,i) - a2(k,j,i))/pcoord->dx3f(k);
      }
    }
  }

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je+1; j++) {
      for (int i=is; i<=ie; i++) {
        pfield->b.x2f(k,j,i) = (a1(k+1,j,i  ) - a1(k,j,i))/pcoord->dx3f(k) -
                               (a3(k  ,j,i+1) - a3(k,j,i))/pcoord->dx1f(i);
      }
    }
  }

  for (int k=ks; k<=ke+1; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        pfield->b.x3f(k,j,i) = (a2(k,j  ,i+1) - a2(k,j,i))/pcoord->dx1f(i) -
                               (a1(k,j+1,i  ) - a1(k,j,i))/pcoord->dx2f(j);
      }
    }
  }

  // Now initialize rest of the cell centered quantities
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x = cos_a2*(pcoord->x1v(i)*cos_a3 + pcoord->x2v(j)*sin_a3) +
                 pcoord->x3v(k)*sin_a2;
        Real sn = std::sin(k_par * x - theta);
        Real cs = fac * std::cos(k_par * x - theta);

        phydro->u(IDN,k,j,i) = den;

        Real mx = den*v_par;
        Real my = -fac*den*v_perp*sn;
        Real mz = -fac*den*v_perp*cs;

        phydro->u(IM1,k,j,i) = mx*cos_a2*cos_a3 - my*sin_a3 - mz*sin_a2*cos_a3;
        phydro->u(IM2,k,j,i) = mx*cos_a2*sin_a3 + my*cos_a3 - mz*sin_a2*sin_a3;
        phydro->u(IM3,k,j,i) = mx*sin_a2                    + mz*cos_a2;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) =
              pres/gm1 +
              0.5*(SQR(0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))) +
                   SQR(0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i))) +
                   SQR(0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i)))) +
              (0.5/den)*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) +
                         SQR(phydro->u(IM3,k,j,i)));
        }
      }
    }
  }

  if (PARTICLES == CHARGED_PAR) {
    // Determine number of particles in the block.
    int npx1_loc(static_cast<int>(
        std::round((block_size.x1max - block_size.x1min) * dx1_reciprocal))),
        npx2_loc(static_cast<int>(std::round(
            (block_size.x2max - block_size.x2min) * dx2_reciprocal))),
        npx3_loc(static_cast<int>(std::round(
            (block_size.x3max - block_size.x3min) * dx3_reciprocal)));
    int npar = ppar->npar = npx1_loc * npx2_loc * npx3_loc;
    if (npar > ppar->nparmax) ppar->UpdateCapacity(npar);

    // Assign the particles.

    if (npar > 0) {
      Real dx1(1.0 / dx1_reciprocal), dx2(1.0 / dx2_reciprocal),
          dx3(1.0 / dx3_reciprocal);
      int ipar = 0;
      for (int k = 0; k < npx3_loc; ++k) {
        Real zp1 = block_size.x3min + (k + 0.5) * dx3;
        for (int j = 0; j < npx2_loc; ++j) {
          Real yp1 = block_size.x2min + (j + 0.5) * dx2;
          for (int i = 0; i < npx1_loc; ++i) {
            Real xp1 = block_size.x1min + (i + 0.5) * dx1;
            ppar->xp(ipar) = xp1;
            ppar->yp(ipar) = yp1;
            ppar->zp(ipar) = zp1;
            ppar->vpx(ipar) = vpx0;
            ppar->vpy(ipar) = vpy0;
            ppar->vpz(ipar) = vpz0;
            ppar->tid(ipar) = 0;
            //ppar->inv_f0(ipar) = 0;
            ++ipar;
          }
        }
      }
    }
  }

  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn Real A1(const Real x1,const Real x2,const Real x3)
//! \brief A1: 1-component of vector potential, using a gauge such that Ax = 0, and Ay,
//! Az are functions of x and y alone.

Real A1(const Real x1, const Real x2, const Real x3) {
  Real x =  x1*cos_a2*cos_a3 + x2*cos_a2*sin_a3 + x3*sin_a2;
  Real y = -x1*sin_a3        + x2*cos_a3;
  Real Ay = fac*(b_perp/k_par)*std::sin(k_par*(x));
  Real Az = (b_perp/k_par)*std::cos(k_par*(x)) + b_par*y;

  return -Ay*sin_a3 - Az*sin_a2*cos_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A2(const Real x1,const Real x2,const Real x3)
//! \brief A2: 2-component of vector potential

Real A2(const Real x1, const Real x2, const Real x3) {
  Real x =  x1*cos_a2*cos_a3 + x2*cos_a2*sin_a3 + x3*sin_a2;
  Real y = -x1*sin_a3        + x2*cos_a3;
  Real Ay = fac*(b_perp/k_par)*std::sin(k_par*(x));
  Real Az = (b_perp/k_par)*std::cos(k_par*(x)) + b_par*y;

  return Ay*cos_a3 - Az*sin_a2*sin_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//! \brief A3: 3-component of vector potential

Real A3(const Real x1, const Real x2, const Real x3) {
  Real x =  x1*cos_a2*cos_a3 + x2*cos_a2*sin_a3 + x3*sin_a2;
  Real y = -x1*sin_a3        + x2*cos_a3;
  Real Az = (b_perp/k_par)*std::cos(k_par*(x)) + b_par*y;

  return Az*cos_a2;
}
} // namespace
