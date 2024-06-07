//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file cpaw1d.cpp
//! \brief Circularly polarized Alfven wave (CPAW) for 1D problems
//!
//! In 1D, the problem is setup along one of the three coordinate axes.  
//!
//! Can be used for [standing/traveling] waves
//! [(problem/v_par=1.0)/(problem/v_par=0.0)]
//!
//! REFERENCE: G. Toth,  "The div(B)=0 constraint in shock capturing MHD codes",
//! JCP,
//!   161, 605 (2000)

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

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

namespace {
// Parameters which define initial solution -- made global so that they can be
// shared with functions A1,2,3 which compute vector potentials
Real den, pres, gm1, b_par, b_perp, v_perp, v_par;
Real fac;
Real lambda, k_par;  // Wavelength, 2*PI/wavelength
}  // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also
//  be used to initialize variables which are global to (and therefore can be
//  passed to) other functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Initialize magnetic field parameters
  // For wavevector along coordinate axes, set desired values of ang_2/ang_3.
  //    For example, for 1D problem use ang_2 = ang_3 = 0.0
  //    For wavevector along grid diagonal, do not input values for ang_2/ang_3.
  // Code below will automatically calculate these imposing periodicity and
  // exactly one wavelength along each grid direction
  b_par = pin->GetReal("problem", "b_par");
  b_perp = pin->GetReal("problem", "b_perp");
  v_par = pin->GetReal("problem", "v_par");
  Real dir =
      pin->GetOrAddReal("problem", "dir", 1);  // right(1)/left(2) polarization
  if (NON_BAROTROPIC_EOS) {
    Real gam = pin->GetReal("hydro", "gamma");
    gm1 = (gam - 1.0);
  }
  pres = pin->GetReal("problem", "pres");
  den = 1.0;

  Real x1size = mesh_size.x1max - mesh_size.x1min;
  Real x2size = mesh_size.x2max - mesh_size.x2min;
  Real x3size = mesh_size.x3max - mesh_size.x3min;

  
  Real x1 = x1size;
  Real x2 = x2size;
  Real x3 = x3size;

  // For lambda choose the smaller of the 3
  lambda = x1;

  // Initialize k_parallel
  k_par = 2.0 * (PI) / lambda;
  v_perp = b_perp / std::sqrt(den);

  if (dir == 1)  // right polarization
    fac = 1.0;
  else  // left polarization
    fac = -1.0;
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//! \brief Compute L1 error in CPAW and output to file
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  if (!pin->GetOrAddBoolean("problem", "compute_error", false)) return;

  // Initialize errors to zero
  Real err[NHYDRO + NFIELD];
  for (int i = 0; i < (NHYDRO + NFIELD); ++i) err[i] = 0.0;

  for (int b = 0; b < nblocal; ++b) {
    MeshBlock *pmb = my_blocks(b);
    //  Compute errors
    for (int k = pmb->ks; k <= pmb->ke; k++) {
      for (int j = pmb->js; j <= pmb->je; j++) {
        for (int i = pmb->is; i <= pmb->ie; i++) {
          Real x = pmb->pcoord->x1v(i);
          Real sn = std::sin(k_par * x);
          Real cs = fac * std::cos(k_par * x);

          err[IDN] += std::abs(den - pmb->phydro->u(IDN, k, j, i));

          Real mx = den * v_par;
          Real my = -fac * den * v_perp * sn;
          Real mz = -fac * den * v_perp * cs;
          Real m1 = mx;
          Real m2 = my ;
          Real m3 = mz;
          err[IM1] += std::abs(m1 - pmb->phydro->u(IM1, k, j, i));
          err[IM2] += std::abs(m2 - pmb->phydro->u(IM2, k, j, i));
          err[IM3] += std::abs(m3 - pmb->phydro->u(IM3, k, j, i));

          Real bx = b_par;
          Real by = b_perp * sn;
          Real bz = b_perp * cs;
          Real b1 = bx;
          Real b2 = by;
          Real b3 = bz;
          err[NHYDRO + IB1] += std::abs(b1 - pmb->pfield->bcc(IB1, k, j, i));
          err[NHYDRO + IB2] += std::abs(b2 - pmb->pfield->bcc(IB2, k, j, i));
          err[NHYDRO + IB3] += std::abs(b3 - pmb->pfield->bcc(IB3, k, j, i));

          if (NON_BAROTROPIC_EOS) {
            Real e0 = pres / gm1 + 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) / den +
                      0.5 * (b1 * b1 + b2 * b2 + b3 * b3);
            err[IEN] += std::abs(e0 - pmb->phydro->u(IEN, k, j, i));
          }
        }
      }
    }
  }

  // normalize errors by number of cells, compute RMS
  for (int i = 0; i < (NHYDRO + NFIELD); ++i) {
    err[i] = err[i] / static_cast<Real>(GetTotalCells());
  }

  Real rms_err = 0.0;
  for (int i = 0; i < (NHYDRO + NFIELD); ++i) rms_err += SQR(err[i]);
  rms_err = std::sqrt(rms_err);

  // open output file and write out errors
  std::string fname;
  fname.assign("cpaw-errors.dat");
  std::stringstream msg;
  FILE *pfile;

  // The file exists -- reopen the file in append mode
  if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
    if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
      msg << "### FATAL ERROR in function [Mesh::UserWorkAfterLoop]"
          << std::endl
          << "Error output file could not be opened" << std::endl;
      ATHENA_ERROR(msg);
    }

    // The file does not exist -- open the file in write mode and add headers
  } else {
    if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
      msg << "### FATAL ERROR in function [Mesh::UserWorkAfterLoop]"
          << std::endl
          << "Error output file could not be opened" << std::endl;
      ATHENA_ERROR(msg);
    }
    std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  RMS-Error  d  M1  M2  M3");
    if (NON_BAROTROPIC_EOS) std::fprintf(pfile, "  E");
    std::fprintf(pfile, "  B1c  B2c  B3c");
    std::fprintf(pfile, "\n");
  }

  // write errors
  std::fprintf(pfile, "%d  %d", mesh_size.nx1, mesh_size.nx2);
  std::fprintf(pfile, "  %d  %d  %e", mesh_size.nx3, ncycle, rms_err);
  std::fprintf(pfile, "  %e  %e  %e  %e", err[IDN], err[IM1], err[IM2],
               err[IM3]);
  if (NON_BAROTROPIC_EOS) std::fprintf(pfile, "  %e", err[IEN]);
  std::fprintf(pfile, "  %e  %e  %e", err[NHYDRO + IB1], err[NHYDRO + IB2],
               err[NHYDRO + IB3]);
  std::fprintf(pfile, "\n");
  std::fclose(pfile);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief circularly polarized Alfven wave problem generator for 1D/2D/3D
//! problems.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Initialize interface fields
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie + 1; i++) {
        pfield->b.x1f(k, j, i) = b_par;
      }
    }
  }

  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je + 1; j++) {
      for (int i = is; i <= ie; i++) {
        pfield->b.x2f(k, j, i) = -b_perp * std::sin(k_par * pcoord->x1v(i));
      }
    }
  }

  for (int k = ks; k <= ke + 1; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        pfield->b.x3f(k, j, i) =
            -fac * b_perp * std::cos(k_par * pcoord->x1v(i));
      }
    }
  }

  // Now initialize rest of the cell centered quantities
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        Real x = pcoord->x1v(i);
        Real sn = std::sin(k_par * x);
        Real cs = fac * std::cos(k_par * x);

        phydro->u(IDN, k, j, i) = den;

        Real mx = den * v_par;
        Real my = -fac * den * v_perp * sn;
        Real mz = -fac * den * v_perp * cs;

        phydro->u(IM1, k, j, i) = mx;
        phydro->u(IM2, k, j, i) = my;
        phydro->u(IM3, k, j, i) = mz;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i) =
              pres / gm1 +
              0.5 *
                  (SQR(0.5 *
                       (pfield->b.x1f(k, j, i) + pfield->b.x1f(k, j, i + 1))) +
                   SQR(0.5 *
                       (pfield->b.x2f(k, j, i) + pfield->b.x2f(k, j + 1, i))) +
                   SQR(0.5 *
                       (pfield->b.x3f(k, j, i) + pfield->b.x3f(k + 1, j, i)))) +
              (0.5 / den) *
                  (SQR(phydro->u(IM1, k, j, i)) + SQR(phydro->u(IM2, k, j, i)) +
                   SQR(phydro->u(IM3, k, j, i)));
        }
      }
    }
  }

  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn Real A1(const Real x1,const Real x2,const Real x3)
//! \brief A1: 1-component of vector potential, using a gauge such that Ax = 0,
//! and Ay, Az are functions of x and y alone.

Real A1(const Real x1, const Real x2, const Real x3) {
  return 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real A2(const Real x1,const Real x2,const Real x3)
//! \brief A2: 2-component of vector potential

Real A2(const Real x1, const Real x2, const Real x3) {
  Real x = x1;
  Real Ay = fac * (b_perp / k_par) * std::sin(k_par * (x));

  return Ay;
}

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//! \brief A3: 3-component of vector potential

Real A3(const Real x1, const Real x2, const Real x3) {
  Real x = x1;
  Real y = x2;
  Real Az = (b_perp / k_par) * std::cos(k_par * (x)) + b_par * y;

  return Az;
}
}  // namespace