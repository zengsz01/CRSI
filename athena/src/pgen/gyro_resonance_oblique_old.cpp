//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file gyro-resonance.cpp
//  \brief Four types circularly polarized Alfven wave (CPAW, left-hand
//  polarized or right-hand polarized, forward or backward) for 1D/2D/3D
//  problems and gyro-resonace instability for charged particles.
//
// The wave always lies on the x-direction.
//
// REFERENCE: G. Toth,  "The div(B)=0 constraint in shock capturing MHD codes",
// JCP, 161, 605 (2000),
// X.-N.Bai+, "Magnetohydrodynamic-Particle-in-Cell Simulations of the
// Cosmic-Ray Streaming Instability: Linear Growth and Quasi-linear Evolution",
// ApJ, 876, 60 (2019)
// J. Squire+, "In-situ switchback formation in the expanding solar wind", ApJ,
// , 891(1), L2 (2020)

// C headers

// C++ headers
#include <algorithm>  // numeric_limits
#include <chrono>
#include <cmath>      // sqrt(), tgamma, abs, pow
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str(), to_string
#include <tuple>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../defs.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../particles/particles.hpp"
#include "../utils/roots.hpp"  // SchroderIterate
#include "../utils/utils.hpp"  // ran2()
#include "../globals.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

namespace {
// Parameters which define initial solution -- made global so that they can be
// shared with functions A1,2,3 which compute vector potentials
// MHD properties
constexpr Real den = 1.0;
Real pres, gam, gm1, b_mag, b_ang, amp, vdft;
int nmax, opt;
std::int64_t iseed_org;
std::vector<Real> phasea, phasef, ampwv, kwv;
// Particle properties
Real dx1_reciprocal, dx2_reciprocal,
    dx3_reciprocal;  // Number density of charged particles
// parameters in the kappa distribution function
Real p0, kappa, kappa_p0sqr_inv, norm_factor;
//Real t_ran_phase_inv;  // time interval for phase randomize.
Real t_adapt_delta_f_inv;  // time interval for phase randomize.
// range of particle distribution, from p0/p_range to p0*p_range
constexpr Real p_range = 500.0;
// accuracy for kappa distribution solution
const Real accuracy =
    std::pow(static_cast<Real>(10),
             -2 * static_cast<int>(std::numeric_limits<Real>::digits10));
std::vector<Real> p_par_bin, f_cum;
Real m_CR, q_CR_over_c;  // mass(charge) fraction of the CRs to background plasma
// analysis properties
bool analysis_enable(false);
int ntbin, npbin;
Real analysis_t_inv, dpbin_inv, ln_p_min, ntbin_inv;
Real *spec;
std::vector<Real> p_bin, mass;
std::string spec_basename;
// Adaptive delta f method
bool adaptive_delta_f_method_enable(false);
Real beta(1.0); // factor to desribe anisotropy level and assuming for yz dir
// functions to solve the kappa - 1.5 from distribution, using the property that
// Utils::BracketAndSolveRoot() can not cross the origin.
struct kappa_delta_functor {
  kappa_delta_functor(Real const &to_find_root_of)
      : a(to_find_root_of) { // Constructor just stores value a to find root of.
  }
  Real operator()(Real const &delta) {
    // CAUTION: need to be this form. If the functor is the reciprocal,
    // Utils::BracketAndSolveRoot() is much more easy to fail
    return SQR(std::tgamma(delta + 2.5)) /
               (0.375 * PI * SQR(delta + 0.5) * SQR(delta + 1.5) *
                std::tgamma(delta + 1.0) * std::tgamma(delta)) -
           a;
  }
 private:
  Real a;  // to be rooted.
};

// functions for the kappa cumulative distribution
struct cdf_kappa_functor {
  // Functor returning both 1st and 2nd derivatives.
  cdf_kappa_functor(Real const &to_find_root_of)
      : a(to_find_root_of),
        factor(4.0 * std::tgamma(kappa + 1.0) / std::tgamma(kappa - 0.5) /
               std::sqrt(PI * kappa) /
               p0) {  // Constructor stores value a to find root of
  }
  std::tuple<Real, Real, Real> operator()(Real const &p) {
    // Return both f(x) and f'(x) and f''(x).
    Real p_sqr(SQR(p));
    Real fx = p * factor * SumSeries(kappa, p_sqr * kappa_p0sqr_inv) -
              a;  // Difference.
    Real dx = 4 * PI * p_sqr * norm_factor *
              std::pow(1.0 + kappa_p0sqr_inv * p_sqr,
                       -kappa - 1);  // 1st derivative
    Real d2x = p * factor * 2 * (SQR(p0) - p_sqr) * kappa *
               pow(1 + p_sqr * kappa_p0sqr_inv, -kappa) /
               SQR(p_sqr + kappa * SQR(p0));  // 2nd derivative
    return std::make_tuple(fx, dx, d2x);      // 'return' fx, dx and d2x.
  }

 private:
  Real a;       // to be rooted.
  Real factor;  // to simplify computing

  // hypergeometric_pFq({0.5, kappa}, {1.5}, z) - hypergeometric_pFq( {0.5,
  // kappa + 1}, {1.5}, z))
  Real SumSeries(const Real &kappa_tmp, const Real &z);
};
// functions for the f_o
// Hypergeometric2F1(a, b, c, z) where z < 1
Real HyperGeoFunc(const Real &a, const Real &b, const Real &c,
                    const Real &z);
Real Mean_functor(const Real &x);
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
  b_mag = pin->GetReal("problem", "b_mag");
  b_ang = pin->GetReal("problem", "b_ang");
  amp = pin->GetReal("problem", "amp");
  vdft = pin->GetReal("problem", "vdft");
  std::int64_t iseed = std::abs(static_cast<std::int64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count()));
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &iseed, 1, MPI_INT64_T, MPI_PROD,
                MPI_COMM_WORLD);
#endif
  iseed_org = pin->GetOrAddInteger("problem", "random_seed", iseed);
  if (iseed_org < 0) iseed_org = iseed;
  iseed = -1 - iseed_org;  // Initialize on the first call to ran2
  ran2(&iseed);
  // initial condition (1: random wave perturbation with input spectrum; 2:
  // single left forward wave; 3 random left forward wave with input spectrum)
  opt = pin->GetOrAddInteger("problem", "option", 1);
  // initial power spectrum index
  Real tur_index = pin->GetOrAddReal("problem", "turbulence_index", -1.0);
  if (NON_BAROTROPIC_EOS) {
    Real gam = pin->GetReal("hydro", "gamma");
    gm1 = (gam - 1.0);
  }
  gam = pin->GetReal("hydro", "gamma");
  pres = pin->GetReal("problem", "pres");

  const Real x1size(mesh_size.x1max - mesh_size.x1min),
      x2size(mesh_size.x2max - mesh_size.x2min),
      x3size(mesh_size.x3max - mesh_size.x3min);

  // For lambda_max choose the smallest of the 3 direction
  Real lambda_max, lambda_min;  // Wavelength, 2*PI/wavelength
  lambda_max = x1size;
  // For lambda_min choose the larger of the 3 direction. len/nx is a compromise
  // choice
  lambda_min = x1size / mesh_size.nx1;
  nmax = mesh_size.nx1;
  lambda_min *= 2.0;
  int n2(nmax / 2);
  Real k_min = 2.0 * PI / lambda_max;
  Real k_max = 2.0 * PI / lambda_min;

  // initialize waves as a superposition of alfven/fast modes
  for (int i = 1; i < nmax; ++i) {
    int i1 = i <= n2 ? i : i - nmax;
    Real k_tmp(SIGN(k_min * i1) * std::min(std::abs(k_min * i1), k_max));
    kwv.push_back(k_tmp);
    phasea.push_back(2.0 * PI * ran2(&iseed));
    phasef.push_back(2.0 * PI * ran2(&iseed));

    // When opt == 1, delete the mode k_tmp == k_max
    if ((k_tmp != k_min && opt == 2) || i == n2 || (k_tmp < k_min && opt == 3))
      ampwv.push_back(0.0);
    else
      ampwv.push_back(amp * std::pow(std::abs(k_tmp) / k_min, 0.5 * tur_index));
  }


  // Initialize charged particles' properties. Marco is prefered due to headers
  if (PARTICLES == CHARGED_PAR) {
    const int nfbin(pin->GetOrAddInteger("problem", "particle_bin", 2));
    // Total number of particles npx1 * npx2 * npx3 * nfbin
    const int npx1((mesh_size.nx1 > 1)
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
    const double dx_reciprocal(dx1_reciprocal * dx2_reciprocal *
                               dx3_reciprocal);

    kappa = pin->GetOrAddReal("problem", "kappa", 1.75);
    
    // default value of p0/m = 300v_A
    p0 = pin->GetOrAddReal("problem", "p0", 300.0 * b_mag / std::sqrt(den));
    kappa_p0sqr_inv = 1.0 / kappa / SQR(p0);
    norm_factor = std::pow(PI * kappa * SQR(p0), -1.5) *
                  std::tgamma(kappa + 1.0) / std::tgamma(kappa - 0.5);
    Real dlnp = 2.0 * std::log(p_range) / static_cast<Real>(nfbin);
    cdf_kappa_functor cdf_kappa(0);  // cumulative kappa distribution function
    for (int i = 0; i <= nfbin; ++i) {
      Real p_par_tmp(p0 / p_range * std::exp(i * dlnp));
      p_par_bin.push_back(p_par_tmp);
      f_cum.push_back(std::get<0>(cdf_kappa(p_par_tmp)));
    }

    m_CR = pin->GetOrAddReal("problem", "m_CR", 0.0);
    const Real qomc(
        pin->GetOrAddReal("problem", "charge_over_mass_over_c", 0.0));
    const Real c_tmp(
        pin->GetOrAddReal("problem", "speed_of_light", HUGE_NUMBER));
    q_CR_over_c = qomc * m_CR;
    /*t_ran_phase_inv =
        1.0 / pin->GetOrAddReal("problem", "random_phase_interval",
                                lambda_max / c_tmp);*/

    // Initialize particles' properties
    for (int i = 0; i < nfbin; ++i) {
      pin->SetReal("particles" + std::to_string(i), "charge_over_mass_over_c",
                   qomc);
      pin->SetReal("particles" + std::to_string(i), "speed_of_light", c_tmp);
      // equivalent particle mass for momentum feedback
      pin->SetReal("particles" + std::to_string(i), "mass",
                   (f_cum[i + 1] - f_cum[i]) * m_CR / dx_reciprocal);
      mass.push_back((f_cum[i + 1] - f_cum[i]) * m_CR / dx_reciprocal);
    }

    // Read adaptive delta f method parameters
    adaptive_delta_f_method_enable =
        pin->GetOrAddBoolean("problem", "adaptive_delta_f_method", false);
    if (adaptive_delta_f_method_enable) {
      beta = pin->GetOrAddReal("problem", "cr_beta", 1.0);
      norm_factor *= SQR(beta);
      t_adapt_delta_f_inv =
          1.0 / pin->GetOrAddReal("problem", "adaptive_delta_f_interval",
                                  lambda_max / c_tmp);
    }
    if (adaptive_delta_f_method_enable && kappa <= 1.5) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "[gyro-resonance]: kappa must be greater than 1.5!" << std::endl;
      ATHENA_ERROR(msg);
    } else if (kappa <= 1.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "[gyro-resonance]: kappa must be greater than 1.0!" << std::endl;
      ATHENA_ERROR(msg);
    } 

    // Initialize analysis properties
    analysis_enable = pin->GetOrAddBoolean("analysis", "enable", false);
    if (analysis_enable || adaptive_delta_f_method_enable)
      spec_basename = pin->GetString("job", "problem_id");
    if (analysis_enable) {
      // Read parameters
      npbin = pin->GetOrAddInteger("analysis", "npbin", nfbin);  // must > 4
      if (npbin <= 4) {
        std::stringstream msg;
        msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]"
            << std::endl
            << "[gyro-resonance]: npbin must be greater than 4!" << std::endl;
        ATHENA_ERROR(msg);
      }
      ntbin = pin->GetOrAddInteger("analysis", "ntbin", 2);
      analysis_t_inv =
          1.0 / pin->GetOrAddReal("analysis", "dt", lambda_max / c_tmp);

      // Initialize
      spec = new Real[npbin * ntbin];
      // add two momentum bins below p0/p_range and two bins above p0*p_range
      dpbin_inv = 0.5 * static_cast<Real>(npbin - 4) / std::log(p_range);
      ln_p_min = std::log(p0 / p_range) - 2.0 / dpbin_inv;
      for (int i = 0; i < npbin; ++i)
        p_bin.push_back(std::exp((i + 0.5) / dpbin_inv + ln_p_min));
      ntbin_inv = 1.0 / static_cast<Real>(ntbin);
    }
  }

  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief Check radius of sphere to make sure it is round
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) { delete spec; }

//========================================================================================
//! \fn ProblemGenerator
//  \brief circularly polarized Alfven wave problem generator for 1D/2D/3D
//  problems.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Initialize interface fields
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie + 1; i++) {
        pfield->b.x1f(k, j, i) = b_mag * std::cos(b_ang);
      }
    }
  }

  for (int k = ks; k <= ke + 1; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        pfield->b.x3f(k, j, i) = b_mag * std::sin(b_ang);
      }
    }
  }

  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je + 1; j++) {
      for (int i = is; i <= ie; i++) {
        for (int n = 0; n < (nmax - 1); ++n) {
          //Alfven Mode perturbation
          pfield->b.x2f(k, j, i) -=
              ampwv[n] * std::sin(kwv[n] * pcoord->x1v(i) - phasea[n]);
        }
      }
    }
  }

  for (int k = ks; k <= ke + 1; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        for (int n = 0; n < (nmax - 1); ++n) {
          //Fast Mode perturbation
          pfield->b.x3f(k, j, i) +=
              ampwv[n] * std::sin(kwv[n] * pcoord->x1v(i) - phasef[n]);
       }
     }
   }
  }

  // Now initialize rest of the cell centered quantities

  // Square of the sound speed
  Real cs_square(gam * pres / den);
  // Phase velocity of the fast mode
  Real vfast(std::sqrt((1 + cs_square + std::sqrt((1+cs_square)*(1+cs_square) - 4 * cs_square * std::cos(b_ang)*std::cos(b_ang)))/2));
  // Ratio of the perturbed vz to perturbed vx
  Real vz_vx_ratio(cs_square * std::sin(2 * b_ang)/(std::sqrt((1+cs_square)*(1+cs_square) - 4 * cs_square * std::cos(b_ang) * std::cos(b_ang)) + 1 - cs_square * std::cos(2 * b_ang)));

  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {

        Real vx(- vdft * std::cos(b_ang)), vy(0.0), vz(- vdft * std::sin(b_ang)), rho(0.0);
        for (int n = 0; n < (nmax - 1); ++n) {
          Real x = pcoord->x1v(i);
          Real amp(std::sqrt(den) * ampwv[n] * SIGN(kwv[n]));

          //perturbed velocity of Alfven Mode
          vy += amp * std::sin(kwv[n] * x - phasea[n]);
          //perutrbed velocity of Fast Mode
          vx += vfast * amp * (std::sin(b_ang) + vz_vx_ratio * std::cos(b_ang)) * std::sin(kwv[n] * x - phasef[n]);
          vz += vfast * amp * (-std::cos(b_ang) + vz_vx_ratio * std::sin(b_ang)) * std::sin(kwv[n] * x - phasef[n]);
          rho += den * ampwv[n] * (std::sin(b_ang) + vz_vx_ratio * std::cos(b_ang)) * std::sin(kwv[n] * x - phasef[n]);          
        }

        phydro->u(IDN, k, j, i) = den + rho;
        phydro->u(IM1, k, j, i) = (den + rho) * vx;
        phydro->u(IM2, k, j, i) = (den + rho) * vy;
        phydro->u(IM3, k, j, i) = (den + rho) * vz;

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

if (PARTICLES == CHARGED_PAR) {
    // Determine number of particles in the block.
  int npx1_loc(static_cast<int>(
      std::round((block_size.x1max - block_size.x1min) * dx1_reciprocal))),
      npx2_loc(static_cast<int>(
          std::round((block_size.x2max - block_size.x2min) * dx2_reciprocal))),
      npx3_loc(static_cast<int>(
          std::round((block_size.x3max - block_size.x3min) * dx3_reciprocal)));
    int npar = ppar->npar =
        npx1_loc * npx2_loc * npx3_loc * ppar->GetTotalTypeNumber();
    if (npar > ppar->nparmax) ppar->UpdateCapacity(npar);

    // random seed, different in meshblock
    std::int64_t iseed(iseed_org + gid);

    // Assign the particles.

    if (npar > 0) {
      Real dx1(1.0 / dx1_reciprocal), dx2(1.0 / dx2_reciprocal),
          dx3(1.0 / dx3_reciprocal);
      int ipar = 0;
      for (std::size_t tid = 0; tid < ppar->GetTotalTypeNumber(); ++tid) {
        const Real p_min(p_par_bin[tid]), p_max(p_par_bin[tid + 1]),
            p_mid(0.5 * (p_min + p_max)), cdf_min(f_cum[tid]),
            cdf_max(f_cum[tid + 1]), cdf_delta(cdf_max - cdf_min);
        for (int k = 0; k < npx3_loc; ++k) {
          Real zp1 = block_size.x3min + (k + 0.5) * dx3;
          for (int j = 0; j < npx2_loc; ++j) {
            Real yp1 = block_size.x2min + (j + 0.5) * dx2;
            for (int i = 0; i < npx1_loc; ++i) {
              Real xp1 = block_size.x1min + (i + 0.5) * dx1;

              Real mytheta(std::acos(2.0 * ran2(&iseed) - 1.0)),
                  myphi(TWO_PI * ran2(&iseed));
              // random generate momentum inside the input bin, following kappa
              // dist
              Real mycdf(cdf_min + cdf_max +
                         cdf_delta * (2.0 * ran2(&iseed) - 1.0));
              mycdf *= 0.5;
              Real myvp(Utils::SchroderIterate(
                  cdf_kappa_functor(mycdf), p_mid, p_min, p_max,
                  static_cast<int>(std::numeric_limits<Real>::digits * 0.4))),
                  sin_mytheta(sin(mytheta));
              if (adaptive_delta_f_method_enable) sin_mytheta /= beta;

              ppar->xp(ipar) = xp1;
              ppar->yp(ipar) = yp1;
              ppar->zp(ipar) = zp1;
              ppar->vpx(ipar) = myvp * std::cos(mytheta);
              ppar->vpy(ipar) = myvp * sin_mytheta * std::cos(myphi);
              ppar->vpz(ipar) = myvp * sin_mytheta * std::sin(myphi);
              ppar->tid(ipar) = tid;
              if (ppar->delta_f_enable)
                ppar->inv_f0(ipar) =
                    1.0 / ppar->PhaseDist(xp1, yp1, zp1, ppar->vpx(ipar),
                                          ppar->vpy(ipar), ppar->vpz(ipar));
              ++ipar;
            }
          }
        }
      }
    }
  }

  return;
}

// Kappa distribution in the phase space
Real ChargedParticles::PhaseDist(const Real &x, const Real &y, const Real &z,
                                 const Real &px, const Real &py,
                                 const Real &pz) const {
  if (adaptive_delta_f_method_enable) {
    Real a1_reciprocal(1.0), a2_reciprocal(1.0), a3_reciprocal(1.0);
    pmy_block->pcoord->CartesianToMeshCoords(1.0, 1.0, 1.0, a1_reciprocal,
                                             a2_reciprocal, a3_reciprocal);
    return norm_factor * a1_reciprocal * a2_reciprocal * a3_reciprocal *
           std::pow(1.0 + kappa_p0sqr_inv *
                            (SQR(px) + SQR(py * beta) + SQR(pz * beta)),
                    -kappa - 1);
  } else if (std::strcmp(COORDINATE_SYSTEM, "expanding") == 0) {
    Real a1(1.0), a2(1.0), a3(1.0);
    pmy_block->pcoord->MeshCoordsToCartesian(1.0, 1.0, 1.0, a1, a2, a3);
    return norm_factor *
           std::pow(1.0 + kappa_p0sqr_inv *
                            (SQR(px * a1) + SQR(py * a2) + SQR(pz * a3)),
                    -kappa - 1);
    /*Real v_sqr(SQR(px) + SQR(py) + SQR(pz)), a_sqr(a2 * a3);
    v_sqr *= kappa_p0sqr_inv;
    return norm_factor * std::pow(1.0 + v_sqr * a_sqr, -kappa - 1.0) *
           HyperGeoFunc(0.5, 1.0 + kappa, 1.5,
                        v_sqr * (a_sqr - 1.0) / (v_sqr + 1.0));*/
  } else
    return norm_factor *
           std::pow(1.0 + kappa_p0sqr_inv * (SQR(px) + SQR(py) + SQR(pz)),
                    -kappa - 1);
}

Real ChargedParticles::ChargeDen0(const Real &x, const Real &y, const Real &z) {
  if (std::strcmp(COORDINATE_SYSTEM, "expanding") == 0) {
    // Assuming at t = 0, a1 = a2 = a3 = 1
    Real a1_reciprocal(1.0), a2_reciprocal(1.0), a3_reciprocal(1.0);
    pmy_block->pcoord->CartesianToMeshCoords(1.0, 1.0, 1.0, a1_reciprocal,
                                             a2_reciprocal, a3_reciprocal);
    return q_CR_over_c * a1_reciprocal * a2_reciprocal * a3_reciprocal;
  }
 
  return q_CR_over_c;
}

//void MeshBlock::UserWorkInLoop() {
//  if (PARTICLES == CHARGED_PAR) ppar->UserWorkInLoop();
//  return;
//}
//
//// Phase randomization
//void Particles::UserWorkInLoop() {
//  if (std::floor(pmy_mesh->time * t_ran_phase_inv) !=
//      std::floor((pmy_mesh->time + pmy_mesh->dt) * t_ran_phase_inv)) {
//    std::int64_t iseed(
//        iseed_org + pmy_block->gid +
//        std::floor((pmy_mesh->time + pmy_mesh->dt) * t_ran_phase_inv));
//
//    // Similar to Boris pusher
//    for (int k = 0; k < npar; ++k) {
//      Real phase_half(PI * ran2(&iseed));
//      Real v_0_x(vpx(k)), v_0_y(vpy(k)), v_0_z(vpz(k));
//      Real temp(SQR(phase_half));
//      // the dot product between b and v
//      Real dot_temp = v_0_x * phase_half;
//      // Rotation due to B field
//      Real v_1_x =- v_0_x * temp + phase_half * dot_temp;
//      Real v_1_y = v_0_z * phase_half - v_0_y * temp;
//      Real v_1_z = -v_0_y * phase_half - v_0_z * temp;
//      temp = 2.0 / (1.0 + temp);  // \frac{2}{1+b^2}
//      // Push final half step due to E field
//      vpx(k) = v_0_x + temp * v_1_x;
//      vpy(k) = v_0_y + temp * v_1_y;
//      vpz(k) = v_0_z + temp * v_1_z;
//    }
//  }
//  return;
//}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::UserWorkInLoop() {
  // Dump particle spectrum
  if (analysis_enable && (std::floor(time * analysis_t_inv) !=
                          std::floor((time + dt) * analysis_t_inv))) {
#pragma ivdep
    std::fill(&spec[0], &spec[npbin * ntbin], 0);

    // Count the particle spectrum in the mesh
    for (int i = 0; i < this->nblocal; ++i)
      this->my_blocks(i)->ppar->AnalysisOutput(spec, ntbin, npbin, dpbin_inv,
                                               ln_p_min);
     
    // sum over all ranks
#ifdef MPI_PARALLEL
    if (Globals::my_rank == 0)
      MPI_Reduce(MPI_IN_PLACE, spec, ntbin * npbin, MPI_ATHENA_REAL, MPI_SUM, 0,
                 MPI_COMM_WORLD);
    else
      MPI_Reduce(spec, spec, ntbin * npbin, MPI_ATHENA_REAL, MPI_SUM, 0,
                 MPI_COMM_WORLD);
#endif  // MPI_PARALLEL

    if (Globals::my_rank == 0) {
      // normalization
      Real norm(1.0 / (mesh_size.x1max - mesh_size.x1min) /
                (mesh_size.x2max - mesh_size.x2min) /
                (mesh_size.x3max - mesh_size.x3min));
      norm *= dpbin_inv * 0.5 * static_cast<Real>(ntbin);
      Real a1(1.0), a2(1.0), a3(1.0), fac(2.0 * PI * norm_factor * m_CR);
      if (std::strcmp(COORDINATE_SYSTEM, "expanding") == 0)
        my_blocks(0)->pcoord->MeshCoordsToCartesian(1.0, 1.0, 1.0, a1, a2, a3);
      norm /= a1 * a2 * a3;
      if (adaptive_delta_f_method_enable) fac /= a1 * a2 * a3;
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i = 0; i < ntbin * npbin; ++i) {
        spec[i] *= norm;
        if (Particles::DeltafEnable()) {
          // TODO(sxc18): CAUTION! This part is hardcoded
          int indp(static_cast<int>(i * ntbin_inv));
          Real myp_sqr(SQR(p_bin[indp]));
          if (adaptive_delta_f_method_enable) {
            int indt(i - indp * ntbin);
            Real mu_sqr(SQR(-1.0 + (indt + 0.5) * 2.0 * ntbin_inv));
            myp_sqr *=  mu_sqr + SQR(beta) * (1.0 - mu_sqr);
            spec[i] += fac * p_bin[indp] * SQR(p_bin[indp]) *
                       std::pow(1.0 + kappa_p0sqr_inv * myp_sqr, -kappa - 1);
          } else if (std::strcmp(COORDINATE_SYSTEM, "expanding") == 0) {
            //int indt(i - indp * ntbin);
            //Real mu_sqr(SQR(-1.0 + (indt + 0.5) * 2.0 * ntbin_inv));
            //// assuming a2 = a3
            //myp_sqr *= SQR(a1) * mu_sqr + SQR(a2) * (1.0 - mu_sqr);
            //spec[i] += fac * p_bin[indp] * SQR(p_bin[indp]) *
            //           std::pow(1.0 + kappa_p0sqr_inv * myp_sqr, -kappa - 1);
            Real a_sqr(a2 * a3);
            spec[i] += fac * p_bin[indp] * SQR(p_bin[indp]) *
                       std::pow(1.0 + kappa_p0sqr_inv * myp_sqr * a_sqr,
                                -kappa - 1.0) *
                       HyperGeoFunc(0.5, 1.0 + kappa, 1.5,
                                    kappa_p0sqr_inv * myp_sqr * (a_sqr - 1.0) /
                                        (kappa_p0sqr_inv * myp_sqr + 1.0));
          } else
            spec[i] += fac * p_bin[indp] * SQR(p_bin[indp]) *
                       std::pow(1.0 + kappa_p0sqr_inv * myp_sqr, -kappa - 1);
        }
      }
       

      // write output file
      std::string fname;
      // Construct file name
      char number[6];
      std::snprintf(number, sizeof(number), "%05d",
                    1 + static_cast<int>(std::floor(time * analysis_t_inv)));
      fname.assign(spec_basename);
      fname.append(".");
      fname.append(number);
      fname.append(".spec");

       // open file for output
      FILE *pfile;
      std::stringstream msg;
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        msg << "### FATAL ERROR in function [Mesh::UserWorkInLoop]" << std::endl
            << "Output file '" << fname << "' could not be opened" << std::endl;
        ATHENA_ERROR(msg);
      }

      // print file header
      std::fprintf(pfile, "# Time:\n");
      std::fprintf(pfile, "%e\n", time);
      std::fprintf(pfile, "# p_min, p_max, nbinp:\n");
      std::fprintf(pfile, "%e %e %d\n", *(p_bin.begin()), *(p_bin.end() - 1),
                   npbin);
      std::fprintf(pfile, "# cost_min, cost_max, ntheta:\n");
      std::fprintf(pfile, "%e %e %d\n", -1.0, 1.0, ntbin);
      std::fprintf(pfile, "\n");

      // loop over all points in spec arrays
      for (int i = 0; i < ntbin * npbin; ++i) {
        int indp(static_cast<int>(i * ntbin_inv));
        int indt(i - indp * ntbin);
        std::fprintf(pfile, "%e %e %e\n", p_bin[indp],
                     -1.0 + (indt + 0.5) * 2.0 * ntbin_inv, spec[i]);
      }
      std::fclose(pfile);
    }
  }

  // Update f_o
  if (adaptive_delta_f_method_enable && Particles::DeltafEnable() &&
      (std::floor(time * t_adapt_delta_f_inv) !=
       std::floor((time + dt) * t_adapt_delta_f_inv))) {

    // statistical value: 0 - mean parallel square; 1 - mean perpendicular
    // square; 2 - mean momentum; 3 - mean momentum square; 4 - normalization
    // factor
    Real stat_local[5], stat_global[5];
    std::fill(&stat_local[0], &stat_local[5], 0);

    // Compute beta first
    for (int i = 0; i < this->nblocal; ++i) {
      stat_local[0] += this->my_blocks(i)->ppar->Statistics(
          [](Real mom, Real mu) { return SQR(mom * mu); });
      stat_local[1] += this->my_blocks(i)->ppar->Statistics(
          [](Real mom, Real mu) { return SQR(mom) * (1.0 - SQR(mu)); });
    }
    // sum over all ranks
#ifdef MPI_PARALLEL
    MPI_Allreduce(stat_local, stat_global, 5, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    std::memcpy(stat_global, stat_local, sizeof(stat_local));
#endif  // MPI_PARALLEL

    // Normailize and add f_o, where expansion factor cancel each other
    Real norm(1.0 / (mesh_size.x1max - mesh_size.x1min) /
              (mesh_size.x2max - mesh_size.x2min) /
              (mesh_size.x3max - mesh_size.x3min)),
        beta_tmp(beta), kappa_tmp(kappa), p0_tmp(p0);
    // Here use beta, kappa and p0 to represent old f_o
    stat_global[0] =
        stat_global[0] * norm + 0.5 * m_CR * SQR(p0) * kappa / (kappa - 1.5);
    stat_global[1] = stat_global[1] * norm +
                     m_CR * SQR(p0) * kappa / (kappa - 1.5) / SQR(beta);

    beta_tmp = std::sqrt(2.0 * stat_global[0] / stat_global[1]);
    const Real beta_sqr(SQR(beta_tmp));

    // Compute kappa and p0 together
    for (int i = 0; i < this->nblocal; ++i) {
      stat_local[2] +=
          this->my_blocks(i)->ppar->Statistics([beta_sqr](Real mom, Real mu) {
        return mom * std::sqrt(SQR(mu) * (1.0 - beta_sqr) + beta_sqr);
      });
      stat_local[3] +=
          this->my_blocks(i)->ppar->Statistics([beta_sqr](Real mom, Real mu) {
        return SQR(mom) * (SQR(mu) * (1.0 - beta_sqr) + beta_sqr);
      });
      stat_local[4] += this->my_blocks(i)->ppar->Statistics(
          [](Real mom, Real mu) { return 1.0; });
    }
    // sum over all ranks
#ifdef MPI_PARALLEL
    MPI_Allreduce(stat_local, stat_global, 5, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    std::memcpy(stat_global, stat_local, sizeof(stat_local));
#endif  // MPI_PARALLEL

    // Normailize and add f_o, where expansion factor cancel each other
    stat_global[2] =
        stat_global[2] * norm + m_CR * p0 * std::tgamma(kappa + 1.0) /
                                    (std::sqrt(PI * kappa) * (kappa - 1.0) *
                                    std::tgamma(kappa - 0.5)) *
                                    Mean_functor(beta / beta_tmp);
    stat_global[3] =
        stat_global[3] * norm + m_CR * kappa * SQR(p0) *
                                    (1.0 + 2.0 * beta_sqr / SQR(beta)) /
                                    (2.0 * kappa - 3.0);
    stat_global[4] = stat_global[4] * norm + m_CR;

    // solve the difference between kappa and 1.5
    kappa_delta_functor to_solve(SQR(stat_global[2]) / stat_global[3] /
                                 stat_global[4]);
    // true due to the function is monotonically increasing
    std::pair<Real, Real> r = Utils::BracketAndSolveRoot(
        to_solve, kappa - 1.5, static_cast<Real>(2.0), true,
        Utils::eps_tolerance<Real>());
    kappa_tmp = r.first * 0.5 + r.second * 0.5 + 1.5;
    p0_tmp = 0.5 * std::sqrt(PI * kappa_tmp) * (kappa_tmp - 1.0) *
             std::tgamma(kappa_tmp - 0.5) / std::tgamma(kappa_tmp + 1.0) *
             stat_global[2] / stat_global[4];

    // Record parameters
    if (Globals::my_rank == 0) {
      std::string fname;
      // Construct file name
      fname.assign(spec_basename);  // using the same
      fname.append(".adap_delta_f_para.hst");

      // open file for output
      FILE *pfile;
      std::stringstream msg;
      if ((pfile = std::fopen(fname.c_str(), "a+")) == nullptr) {
        msg << "### FATAL ERROR in function [Mesh::UserWorkInLoop]" << std::endl
            << "Output file '" << fname << "' could not be opened" << std::endl;
        ATHENA_ERROR(msg);
      }
      std::fprintf(pfile, "time= %e: %e %e %e\n", time, beta_tmp, kappa_tmp, p0_tmp);
      std::fclose(pfile);
    }

    // update and re-calculate parameters
    beta = beta_tmp;
    kappa = kappa_tmp;
    p0 = p0_tmp;
    kappa_p0sqr_inv = 1.0 / kappa / SQR(p0);
    norm_factor = std::pow(PI * kappa * SQR(p0), -1.5) * beta_sqr *
                  std::tgamma(kappa + 1.0) / std::tgamma(kappa - 0.5);
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AnalysisOutput(Real *spec, const int &ntbin, const
//! int &npbin, const Real &dpbin_inv, const Real &ln_p_min) const
//  \brief returns the particle spectrum, spec, with given parameter about p
//  axis (npbin, dpbin_inv, ln_p_min) and pitch angle axis (ntbin).

void Particles::AnalysisOutput(Real *spec_tmp, const int &ntbin_tmp,
                               const int &npbin_tmp, const Real &dpbin_inv_tmp,
                               const Real &ln_p_min_tmp) const {
  // TODO(sxc18): This code is hardcoded and could be vectorized potentially.

  // Loop over all the particles
  for (int k = 0; k < npar; ++k) {
    Real myp(std::sqrt(SQR(vpx(k)) + SQR(vpy(k)) + SQR(vpz(k))));
    int indp(static_cast<int>((std::log(myp) - ln_p_min_tmp) * dpbin_inv_tmp));
    if (indp >= npbin_tmp) continue;
    if (indp < 0) continue;
    Real costheta(vpx(k) / myp), mym(mass[Particles::tid(k)]);
    int indt(std::min(
        static_cast<int>(static_cast<int>(0.5 * ntbin_tmp * (1.0 + costheta))),
        ntbin_tmp - 1));
    if (delta_f_enable)
      mym *= 1.0 -
             PhaseDist(xp(k), yp(k), zp(k), vpx(k), vpy(k), vpz(k)) * inv_f0(k);

    spec_tmp[indt + indp * ntbin_tmp] += mym;
  }
}


namespace {
// z must be larger than 0
Real cdf_kappa_functor::SumSeries(const Real &kappa_tmp, const Real &z) {
  // if (z <= 0) {
  //  std::stringstream msg;
  //  msg << "### FATAL ERROR in function [cdf_kappa_functor::SumSeries]"
  //      << std::endl
  //      << "[gyro-resonance]: z must be larger than 0!" << std::endl;
  //  ATHENA_ERROR(msg);
  //}
  Real sum(0), n(0);
  // Pfaff transformations
  const Real z_plus_1_mul_kappa_inv(1.0 / (1 + z) / kappa_tmp),
      z_div_z_plus_1(z / (1.0 + z));
  Real factor(std::pow(1 + z, -kappa_tmp)), term(1.0 - 1.0 / (1 + z));
  Real series(factor * term);
  do {
    sum += series;
    factor *= z_div_z_plus_1 * (kappa_tmp + n) / (1.5 + n);
    term -= z_plus_1_mul_kappa_inv;
    ++n;
    series = factor * term;
  } while (std::abs(series) >= std::abs(sum) * accuracy);

  return sum;
}

// z must be smaller than 1
Real HyperGeoFunc(const Real &a, const Real &b, const Real &c,
                    const Real &z) {
  Real sum(0), n(0);

  if (std::abs(z) < 1.0) {
    Real factor(1.0);
    Real series(factor);
    do {
      sum += series;
      factor *= z * (a + n) * (b + n) / (n + 1.0) / (c + n);
      ++n;
      series = factor;
    } while (std::abs(series) >= std::abs(sum) * accuracy);
  } else if (z < 0) {
    // Pfaff transformations
    sum = std::pow(1 - z, -a) * HyperGeoFunc(a, c - b, c, z / (z - 1.0));
  }

  return sum;
}

Real Mean_functor(const Real &x) {
  const Real x_sqr(SQR(x));
  if (x_sqr > 1.0)
    return 1.0 +
           std::asinh(std::sqrt(x_sqr - 1.0)) / x_sqr / std::sqrt(x_sqr - 1.0);
  else if (x_sqr < 1.0)
    return 1.0 +
           std::asin(std::sqrt(1.0 - x_sqr)) / x_sqr / std::sqrt(1.0 - x_sqr);
  else
    return 2.0;
}

}  // namespace
