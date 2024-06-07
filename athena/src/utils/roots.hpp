#ifndef UTILS_ROOTS_HPP_
#define UTILS_ROOTS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file roots.hpp
//  \brief a copy for the roots.hpp in the Boost library, offering three
//  functions for roots:
//  -- Utils::Toms748Solve(), (unsuggested)
//  -- Utils::BracketAndSolveRoot(),
//  -- Utils::Bisect(),
//  -- Utils::NewtonRaphsonIterate(),
//  -- Utils::HalleyIterate(),
//  -- Utils::SchroderIterate().
//  The functor to be calculated must follow the type like in
//  src/pgen/gyro_resonace.cpp:cdf_kappa_functor or the examples from the boost
//  library.
//  Here is the original license: (C) Copyright John Maddock 2006. Use,
//  modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// C headers
#include <assert.h>  // asseert

// C++ headers
#include <cmath>
#include <iostream>  // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <tuple>
#include <utility>

// Athena++ headers
#include "../athena.hpp"
#include "../defs.hpp"

namespace Utils {
template <class T>
class eps_tolerance {
 public:
  eps_tolerance() { eps = 4 * std::numeric_limits<T>::epsilon(); }
  eps_tolerance(unsigned bits) {
    eps = (std::max)(T(std::ldexp(1.0F, 1 - bits)),
                     T(4 * std::numeric_limits<T>::epsilon()));
  }
  bool operator()(const T& a, const T& b) {
    return std::abs(a - b) <= (eps * (std::min)(std::abs(a), std::abs(b)));
  }

 private:
  T eps;
};

struct equal_floor {
  equal_floor() {}
  template <class T>
  bool operator()(const T& a, const T& b) {
    return std::floor(a) == std::floor(b);
  }
};

struct equal_ceil {
  equal_ceil() {}
  template <class T>
  bool operator()(const T& a, const T& b) {
    return std::ceil(a) == std::ceil(b);
  }
};

struct equal_nearest_integer {
  equal_nearest_integer() {}
  template <class T>
  bool operator()(const T& a, const T& b) {
    return std::floor(a + 0.5f) == std::floor(b + 0.5f);
  }
};

namespace Detail {

template <class F, class T>
void Bracket(F f, T& a, T& b, T c, T& fa, T& fb, T& d, T& fd) {
  //
  // Given a point c inside the existing enclosing interval
  // [a, b] sets a = c if f(c) == 0, otherwise finds the new
  // enclosing interval: either [a, c] or [c, b] and sets
  // d and fd to the point that has just been removed from
  // the interval.  In other words d is the third best guess
  // to the root.
  //
  T tol = std::numeric_limits<T>::epsilon() * 2;
  //
  // If the interval [a,b] is very small, or if c is too close
  // to one end of the interval then we need to adjust the
  // location of c accordingly:
  //
  if ((b - a) < 2 * tol * a) {
    c = a + (b - a) / 2;
  } else if (c <= a + std::abs(a) * tol) {
    c = a + std::abs(a) * tol;
  } else if (c >= b - std::abs(b) * tol) {
    c = b - std::abs(b) * tol;
  }
  //
  // OK, lets invoke f(c):
  //
  T fc = f(c);
  //
  // if we have a zero then we have an exact solution to the root:
  //
  if (fc == 0) {
    a = c;
    fa = 0;
    d = 0;
    fd = 0;
    return;
  }
  //
  // Non-zero fc, update the interval:
  //
  if (SIGN(fa) * SIGN(fc) < 0) {
    d = b;
    fd = fb;
    b = c;
    fb = fc;
  } else {
    d = a;
    fd = fa;
    a = c;
    fa = fc;
  }
}

template <class T>
inline T SafeDiv(T num, T denom, T r) {
  //
  // return num / denom without overflow,
  // return r if overflow would occur.
  //

  if (std::abs(denom) < 1) {
    if (std::abs(denom * std::numeric_limits<T>::max()) <= std::abs(num))
      return r;
  }
  return num / denom;
}

template <class T>
inline T SecantInterpolate(const T& a, const T& b, const T& fa, const T& fb) {
  //
  // Performs standard secant interpolation of [a,b] given
  // function evaluations f(a) and f(b).  Performs a bisection
  // if secant interpolation would leave us very close to either
  // a or b.  Rationale: we only call this function when at least
  // one other form of interpolation has already failed, so we know
  // that the function is unlikely to be smooth with a root very
  // close to a or b.
  //

  T tol = std::numeric_limits<T>::epsilon() * 5;
  T c = a - (fa / (fb - fa)) * (b - a);
  if ((c <= a + std::abs(a) * tol) || (c >= b - std::abs(b) * tol))
    return (a + b) / 2;
  return c;
}

template <class T>
T QuadraticInterpolate(const T& a, const T& b, T const& d, const T& fa,
                       const T& fb, T const& fd, unsigned count) {
  //
  // Performs quadratic interpolation to determine the next point,
  // takes count Newton steps to find the location of the
  // quadratic polynomial.
  //
  // Point d must lie outside of the interval [a,b], it is the third
  // best approximation to the root, after a and b.
  //
  // Note: this does not guarantee to find a root
  // inside [a, b], so we fall back to a secant step should
  // the result be out of range.
  //
  // Start by obtaining the coefficients of the quadratic polynomial:
  //
  T B = SafeDiv(T(fb - fa), T(b - a), std::numeric_limits<T>::max());
  T A = SafeDiv(T(fd - fb), T(d - b), std::numeric_limits<T>::max());
  A = SafeDiv(T(A - B), T(d - a), T(0));

  if (A == 0) {
    // failure to determine coefficients, try a secant step:
    return SecantInterpolate(a, b, fa, fb);
  }
  //
  // Determine the starting point of the Newton steps:
  //
  T c;
  if (SIGN(A) * SIGN(fa) > 0) {
    c = a;
  } else {
    c = b;
  }
  //
  // Take the Newton steps:
  //
  for (unsigned i = 1; i <= count; ++i) {
    // c -= SafeDiv(B * c, (B + A * (2 * c - a - b)), 1 + c - a);
    c -= SafeDiv(T(fa + (B + A * (c - b)) * (c - a)),
                 T(B + A * (2 * c - a - b)), T(1 + c - a));
  }
  if ((c <= a) || (c >= b)) {
    // Oops, failure, try a secant step:
    c = SecantInterpolate(a, b, fa, fb);
  }
  return c;
}

template <class T>
T CubicInterpolate(const T& a, const T& b, const T& d, const T& e, const T& fa,
                   const T& fb, const T& fd, const T& fe) {
  //
  // Uses inverse cubic interpolation of f(x) at points
  // [a,b,d,e] to obtain an approximate root of f(x).
  // Points d and e lie outside the interval [a,b]
  // and are the third and forth best approximations
  // to the root that we have found so far.
  //
  // Note: this does not guarantee to find a root
  // inside [a, b], so we fall back to quadratic
  // interpolation in case of an erroneous result.
  //

  T q11 = (d - e) * fd / (fe - fd);
  T q21 = (b - d) * fb / (fd - fb);
  T q31 = (a - b) * fa / (fb - fa);
  T d21 = (b - d) * fd / (fd - fb);
  T d31 = (a - b) * fb / (fb - fa);

  T q22 = (d21 - q11) * fb / (fe - fb);
  T q32 = (d31 - q21) * fa / (fd - fa);
  T d32 = (d31 - q21) * fd / (fd - fa);
  T q33 = (d32 - q22) * fa / (fe - fa);
  T c = q31 + q32 + q33 + a;

  if ((c <= a) || (c >= b))
    // Out of bounds step, fall back to quadratic interpolation:
    c = QuadraticInterpolate(a, b, d, fa, fb, fd, 3);

  return c;
}

}  // namespace Detail

template <class F, class T, class Tol>
std::pair<T, T> Toms748Solve(
    F f, const T& ax, const T& bx, const T& fax, const T& fbx, Tol tol,
    std::uint64_t& max_iter) noexcept(std::is_floating_point<T>::
                                          value&& noexcept(std::declval<F>()(
                                              std::declval<T>()))) {
  //
  // Main entry point and logic for Toms Algorithm 748
  // root finder.
  //

  //
  // Sanity check - are we allowed to iterate at all?
  //
  if (max_iter == 0) return std::make_pair(ax, bx);

  std::uint64_t count = max_iter;
  T a, b, fa, fb, c, u, fu, a0, b0, d, fd, e, fe;
  static const T mu = 0.5f;

  // initialise a, b and fa, fb:
  a = ax;
  b = bx;
  if (a >= b) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Utils::Toms748Solve]" << std::endl
        << "[roots]: Parameters a and b out of order" << std::endl;
    ATHENA_ERROR(msg);
    return std::make_pair(a, a);
  }

  fa = fax;
  fb = fbx;

  if (tol(a, b) || (fa == 0) || (fb == 0)) {
    max_iter = 0;
    if (fa == 0)
      b = a;
    else if (fb == 0)
      a = b;
    return std::make_pair(a, b);
  }

  if (SIGN(fa) * SIGN(fb) > 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Utils::Toms748Solve]" << std::endl
        << "[roots]: Parameters a and b do not bracket the root" << std::endl;
    ATHENA_ERROR(msg);
    return std::make_pair(a, a);
  }

  // dummy value for fd, e and fe:
  fe = e = fd = 1e5F;

  if (fa != 0) {
    //
    // On the first step we take a secant step:
    //
    c = Detail::SecantInterpolate(a, b, fa, fb);
    Detail::Bracket(f, a, b, c, fa, fb, d, fd);
    --count;

    if (count && (fa != 0) && !tol(a, b)) {
      //
      // On the second step we take a quadratic interpolation:
      //
      c = Detail::QuadraticInterpolate(a, b, d, fa, fb, fd, 2);
      e = d;
      fe = fd;
      Detail::Bracket(f, a, b, c, fa, fb, d, fd);
      --count;
    }
  }

  while (count && (fa != 0) && !tol(a, b)) {
    // save our brackets:
    a0 = a;
    b0 = b;
    //
    // Starting with the third step taken
    // we can use either quadratic or cubic interpolation.
    // Cubic interpolation requires that all four function values
    // fa, fb, fd, and fe are distinct, should that not be the case
    // then variable prof will get set to true, and we'll end up
    // taking a quadratic step instead.
    //
    T min_diff = std::numeric_limits<T>::min() * 32;
    bool prof =
        (std::abs(fa - fb) < min_diff) || (std::abs(fa - fd) < min_diff) ||
        (std::abs(fa - fe) < min_diff) || (std::abs(fb - fd) < min_diff) ||
        (std::abs(fb - fe) < min_diff) || (std::abs(fd - fe) < min_diff);
    if (prof)
      c = Detail::QuadraticInterpolate(a, b, d, fa, fb, fd, 2);
    else
      c = Detail::CubicInterpolate(a, b, d, e, fa, fb, fd, fe);

    //
    // re-bracket, and check for termination:
    //
    e = d;
    fe = fd;
    Detail::Bracket(f, a, b, c, fa, fb, d, fd);
    if ((0 == --count) || (fa == 0) || tol(a, b)) break;
    //
    // Now another interpolated step:
    //
    prof = (std::abs(fa - fb) < min_diff) || (std::abs(fa - fd) < min_diff) ||
           (std::abs(fa - fe) < min_diff) || (std::abs(fb - fd) < min_diff) ||
           (std::abs(fb - fe) < min_diff) || (std::abs(fd - fe) < min_diff);
    if (prof)
      c = Detail::QuadraticInterpolate(a, b, d, fa, fb, fd, 3);
    else
      c = Detail::CubicInterpolate(a, b, d, e, fa, fb, fd, fe);

    //
    // Bracket again, and check termination condition, update e:
    //
    Detail::Bracket(f, a, b, c, fa, fb, d, fd);
    if ((0 == --count) || (fa == 0) || tol(a, b)) break;
    //
    // Now we take a double-length secant step:
    //
    if (std::abs(fa) < std::abs(fb)) {
      u = a;
      fu = fa;
    } else {
      u = b;
      fu = fb;
    }
    c = u - 2 * (fu / (fb - fa)) * (b - a);
    if (std::abs(c - u) > (b - a) / 2) {
      c = a + (b - a) / 2;
    }
    //
    // Bracket again, and check termination condition:
    //
    e = d;
    fe = fd;
    Detail::Bracket(f, a, b, c, fa, fb, d, fd);
    if ((0 == --count) || (fa == 0) || tol(a, b)) break;
    //
    // And finally... check to see if an additional bisection step is
    // to be taken, we do this if we're not converging fast enough:
    //
    if ((b - a) < mu * (b0 - a0)) continue;
    //
    // bracket again on a bisection:
    //
    e = d;
    fe = fd;
    Detail::Bracket(f, a, b, T(a + (b - a) / 2), fa, fb, d, fd);
    --count;
  }  // while loop

  max_iter -= count;
  if (fa == 0) {
    b = a;
  } else if (fb == 0) {
    a = b;
  }
  return std::make_pair(a, b);
}

template <class F, class T, class Tol>
inline std::pair<T, T> Toms748Solve(F f, const T& ax, const T& bx, Tol tol,
                                    std::uint64_t& max_iter) {
  if (max_iter <= 2) return std::make_pair(ax, bx);
  max_iter -= 2;
  std::pair<T, T> r = Toms748Solve(f, ax, bx, f(ax), f(bx), tol, max_iter);
  max_iter += 2;
  return r;
}

template <class F, class T, class Tol>
inline std::pair<T, T> Toms748Solve(F f, const T& ax, const T& bx, Tol tol) {
  std::uint64_t m = (std::numeric_limits<std::uint64_t>::max)();
  return Toms748Solve(f, ax, bx, tol, m);
}

template <class F, class T, class Tol>
std::pair<T, T> BracketAndSolveRoot(
    F f, const T& guess, T factor, bool rising, Tol tol,
    std::uint64_t& max_iter) noexcept(std::is_floating_point<T>::
                                          value&& noexcept(std::declval<F>()(
                                              std::declval<T>()))) {
  //
  // Set up initial brackets:
  //
  T a = guess;
  T b = a;
  T fa = f(a);
  T fb = fa;
  //
  // Set up invocation count:
  //
  std::uint64_t count = max_iter - 1;

  int step = 32;

  if ((fa < 0) == (guess < 0 ? !rising : rising)) {
    //
    // Zero is to the right of b, so walk upwards
    // until we find it:
    //
    while (SIGN(fb) == SIGN(fa)) {
      if (count == 0) {
        std::stringstream msg;
        msg << "### FATAL ERROR in function [Utils::BracketAndSolveRoot]"
            << std::endl
            << "[roots]: Unable to bracket root, last nearest value was " << b
            << std::endl;
        ATHENA_ERROR(msg);
        return std::make_pair(b, b);
      }
      //
      // Heuristic: normally it's best not to increase the step sizes as we'll
      // just end up with a really wide range to search for the root.  However,
      // if the initial guess was *really* bad then we need to speed up the
      // search otherwise we'll take forever if we're orders of magnitude out.
      // This happens most often if the guess is a small value (say 1) and the
      // result we're looking for is close to std::numeric_limits<T>::min().
      //
      if ((max_iter - count) % step == 0) {
        factor *= 2;
        if (step > 1) step /= 2;
      }
      //
      // Now go ahead and move our guess by "factor":
      //
      a = b;
      fa = fb;
      b *= factor;
      fb = f(b);
      --count;
    }
  } else {
    //
    // Zero is to the left of a, so walk downwards
    // until we find it:
    //
    while (SIGN(fb) == SIGN(fa)) {
      if (std::abs(a) < std::numeric_limits<T>::min()) {
        // Escape route just in case the answer is zero!
        max_iter -= count;
        max_iter += 1;
        return a > 0 ? std::make_pair(T(0), T(a)) : std::make_pair(T(a), T(0));
      }
      if (count == 0) {
        std::stringstream msg;
        msg << "### FATAL ERROR in function [Utils::BracketAndSolveRoot]"
            << std::endl
            << "[roots]: Unable to bracket root, last nearest value was " << a
            << std::endl;
        ATHENA_ERROR(msg);
        return std::make_pair(a, a);
      }

      //
      // Heuristic: normally it's best not to increase the step sizes as we'll
      // just end up with a really wide range to search for the root.  However,
      // if the initial guess was *really* bad then we need to speed up the
      // search otherwise we'll take forever if we're orders of magnitude out.
      // This happens most often if the guess is a small value (say 1) and the
      // result we're looking for is close to std::numeric_limits<T>::min().
      //
      if ((max_iter - count) % step == 0) {
        factor *= 2;
        if (step > 1) step /= 2;
      }
      //
      // Now go ahead and move are guess by "factor":
      //
      b = a;
      fb = fa;
      a /= factor;
      fa = f(a);
      --count;
    }
  }
  max_iter -= count;
  max_iter += 1;
  std::pair<T, T> r =
      Toms748Solve(f, (a < 0 ? b : a), (a < 0 ? a : b), (a < 0 ? fb : fa),
                   (a < 0 ? fa : fb), tol, count);
  max_iter += count;
  return r;
}

template <class F, class T, class Tol>
inline std::pair<T, T> BracketAndSolveRoot(F f, const T& guess, T factor,
                                           bool rising, Tol tol) {
  std::uint64_t m = (std::numeric_limits<std::uint64_t>::max)();
  return BracketAndSolveRoot(f, guess, factor, rising, tol, m);
}

template <class F, class T, class Tol>
std::pair<T, T>
Bisect(F f, T min, T max, Tol tol, std::uint64_t& max_iter) noexcept(
    std::is_floating_point<T>::value&& noexcept(
        std::declval<F>()(std::declval<T>()))) {
  T fmin = f(min);
  T fmax = f(max);
  if (fmin == 0) {
    max_iter = 2;
    return std::make_pair(min, min);
  }
  if (fmax == 0) {
    max_iter = 2;
    return std::make_pair(max, max);
  }

  //
  // Error checking:
  //
  static const char* function = "boost::math::tools::bisect<%1%>";
  if (min >= max) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Utils::Bisect]" << std::endl
        << "[roots]: Arguments in wrong order" << std::endl;
    ATHENA_ERROR(msg);
    return std::make_pair(min, min);
  }
  if (fmin * fmax >= 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Utils::Bisect]" << std::endl
        << "[roots]: No change of sign, either there is "
           "no root to find, or there are multiple roots in the interval"
        << std::endl;
    ATHENA_ERROR(msg);
    return std::make_pair(fmin, fmin);
  }

  //
  // Three function invocations so far:
  //
  std::uint64_t count = max_iter;
  if (count < 3)
    count = 0;
  else
    count -= 3;

  while (count && (0 == tol(min, max))) {
    T mid = (min + max) / 2;
    T fmid = f(mid);
    if ((mid == max) || (mid == min)) break;
    if (fmid == 0) {
      min = max = mid;
      break;
    } else if (sign(fmid) * sign(fmin) < 0) {
      max = mid;
    } else {
      min = mid;
      fmin = fmid;
    }
    --count;
  }

  max_iter -= count;

  return std::make_pair(min, max);
}

namespace Detail {

namespace Dummy {

template <int n, class T>
typename T::value_type get(const T&) noexcept(std::is_floating_point<T>::value);
}

template <class Tuple, class T>
void UnpackTuple(const Tuple& t, T& a,
                 T& b) noexcept(std::is_floating_point<T>::value) {
  using Dummy::get;
  // Use ADL to find the right overload for get:
  a = get<0>(t);
  b = get<1>(t);
}
template <class Tuple, class T>
void UnpackTuple(const Tuple& t, T& a, T& b,
                 T& c) noexcept(std::is_floating_point<T>::value) {
  using Dummy::get;
  // Use ADL to find the right overload for get:
  a = get<0>(t);
  b = get<1>(t);
  c = get<2>(t);
}

template <class Tuple, class T>
inline void Unpack_0(const Tuple& t,
                     T& val) noexcept(std::is_floating_point<T>::value) {
  using Dummy::get;
  // Rely on ADL to find the correct overload of get:
  val = get<0>(t);
}

template <class T, class U, class V>
inline void UnpackTuple(const std::pair<T, U>& p, V& a,
                        V& b) noexcept(std::is_floating_point<T>::value) {
  a = p.first;
  b = p.second;
}
template <class T, class U, class V>
inline void Unpack_0(const std::pair<T, U>& p,
                     V& a) noexcept(std::is_floating_point<T>::value) {
  a = p.first;
}

template <class F, class T>
void HandleZeroDerivative(
    F f, T& last_f0, const T& f0, T& delta, T& result, T& guess, const T& min,
    const T& max) noexcept(std::is_floating_point<T>::
                               value&& noexcept(
                                   std::declval<F>()(std::declval<T>()))) {
  if (last_f0 == 0) {
    // this must be the first iteration, pretend that we had a
    // previous one at either min or max:
    if (result == min) {
      guess = max;
    } else {
      guess = min;
    }
    Unpack_0(f(guess), last_f0);
    delta = guess - result;
  }
  if (SIGN(last_f0) * SIGN(f0) < 0) {
    // we've crossed over so move in opposite direction to last step:
    if (delta < 0) {
      delta = (result - min) / 2;
    } else {
      delta = (result - max) / 2;
    }
  } else {
    // move in same direction as last step:
    if (delta < 0) {
      delta = (result - max) / 2;
    } else {
      delta = (result - min) / 2;
    }
  }
}

}  // namespace Detail

template <class F, class T>
T NewtonRaphsonIterate(
    F f, T guess, T min, T max, int digits,
    std::uint64_t& max_iter) noexcept(std::is_floating_point<T>::
                                          value&& noexcept(std::declval<F>()(
                                              std::declval<T>()))) {
  if (min > max) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Utils::NewtonRaphsonIterate]"
        << std::endl
        << "[roots]: Range arguments in wrong order" << std::endl;
    ATHENA_ERROR(msg);
  }

  T f0(0), f1, last_f0(0);
  T result = guess;

  T factor = static_cast<T>(std::ldexp(1.0, 1 - digits));
  T delta = (std::numeric_limits<T>::max)();
  T delta1 = (std::numeric_limits<T>::max)();
  T delta2 = (std::numeric_limits<T>::max)();

  //
  // We use these to sanity check that we do actually bracket a root,
  // we update these to the function value when we update the endpoints
  // of the range.  Then, provided at some point we update both endpoints
  // checking that max_range_f * min_range_f <= 0 verifies there is a root
  // to be found somewhere.  Note that if there is no root, and we approach
  // a local minima, then the derivative will go to zero, and hence the next
  // step will jump out of bounds (or at least past the minima), so this
  // check *should* happen in pathological cases.
  //
  T max_range_f = 0;
  T min_range_f = 0;

  std::uint64_t count(max_iter);

  do {
    last_f0 = f0;
    delta2 = delta1;
    delta1 = delta;
    Detail::UnpackTuple(f(result), f0, f1);
    --count;
    if (0 == f0) break;
    if (f1 == 0) {
      // Oops zero derivative!!!
      Detail::HandleZeroDerivative(f, last_f0, f0, delta, result, guess, min,
                                   max);
    } else {
      delta = f0 / f1;
    }

    if (std::abs(delta * 2) > std::abs(delta2)) {
      // Last two steps haven't converged.
      T shift = (delta > 0) ? (result - min) / 2 : (result - max) / 2;
      if ((result != 0) && (std::abs(shift) > std::abs(result))) {
        delta = SIGN(delta) * std::abs(result) *
                1.1f;  // Protect against huge jumps!
        // delta = SIGN(delta) * result; // Protect against huge jumps! Failed
        // for negative result. https://github.com/boostorg/math/issues/216
      } else
        delta = shift;
      // reset delta1/2 so we don't take this branch next time round:
      delta1 = 3 * delta;
      delta2 = 3 * delta;
    }
    guess = result;
    result -= delta;
    if (result <= min) {
      delta = 0.5F * (guess - min);
      result = guess - delta;
      if ((result == min) || (result == max)) break;
    } else if (result >= max) {
      delta = 0.5F * (guess - max);
      result = guess - delta;
      if ((result == min) || (result == max)) break;
    }
    // Update brackets:
    if (delta > 0) {
      max = guess;
      max_range_f = f0;
    } else {
      min = guess;
      min_range_f = f0;
    }
    //
    // Sanity check that we bracket the root:
    //
    if (max_range_f * min_range_f > 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [Utils::NewtonRaphsonIterate]"
          << std::endl
          << "[roots]: There appears to be no root to be found" << std::endl;
      ATHENA_ERROR(msg);
    }
  } while (count && (std::abs(result * factor) < std::abs(delta)));

  max_iter -= count;

  return result;
}

template <class F, class T>
inline T NewtonRaphsonIterate(F f, T guess, T min, T max, int digits) noexcept(
    std::is_floating_point<T>::value&& noexcept(
        std::declval<F>()(std::declval<T>()))) {
  std::uint64_t m = (std::numeric_limits<std::uint64_t>::max)();
  return NewtonRaphsonIterate(f, guess, min, max, digits, m);
}

namespace Detail {

struct HalleyStep {
  template <class T>
  static T step(const T& /*x*/, const T& f0, const T& f1,
                const T& f2) noexcept(std::is_floating_point<T>::value) {
    T denom = 2 * f0;
    T num = 2 * f1 - f0 * (f2 / f1);
    T delta;

    if ((std::abs(num) < 1) &&
        (std::abs(denom) >= std::abs(num) * (std::numeric_limits<T>::max)())) {
      // possible overflow, use Newton step:
      delta = f0 / f1;
    } else
      delta = denom / num;
    return delta;
  }
};

template <class F, class T>
T BracketRootTowardsMin(
    F f, T guess, const T& f0, T& min, T& max,
    std::uint64_t& count) noexcept(std::is_floating_point<T>::
                                       value&& noexcept(std::declval<F>()(
                                           std::declval<T>())));

template <class F, class T>
T BracketRootTowardsMax(
    F f, T guess, const T& f0, T& min, T& max,
    std::uint64_t& count) noexcept(std::is_floating_point<T>::
                                       value&& noexcept(std::declval<F>()(
                                           std::declval<T>()))) {
  //
  // Move guess towards max until we bracket the root, updating min and max as
  // we go:
  //
  T guess0 = guess;
  T multiplier = 2;
  T f_current = f0;
  if (std::abs(min) < std::abs(max)) {
    while (--count && ((f_current < 0) == (f0 < 0))) {
      min = guess;
      guess *= multiplier;
      if (guess > max) {
        guess = max;
        f_current = -f_current;  // There must be a change of sign!
        break;
      }
      multiplier *= 2;
      Unpack_0(f(guess), f_current);
    }
  } else {
    //
    // If min and max are negative we have to divide to head towards max:
    //
    while (--count && ((f_current < 0) == (f0 < 0))) {
      min = guess;
      guess /= multiplier;
      if (guess > max) {
        guess = max;
        f_current = -f_current;  // There must be a change of sign!
        break;
      }
      multiplier *= 2;
      Unpack_0(f(guess), f_current);
    }
  }

  if (count) {
    max = guess;
    if (multiplier > 16)
      return (guess0 - guess) +
             BracketRootTowardsMin(f, guess, f_current, min, max, count);
  }
  return guess0 - (max + min) / 2;
}

template <class F, class T>
T BracketRootTowardsMin(
    F f, T guess, const T& f0, T& min, T& max,
    std::uint64_t& count) noexcept(std::is_floating_point<T>::
                                       value&& noexcept(std::declval<F>()(
                                           std::declval<T>()))) {
  //
  // Move guess towards min until we bracket the root, updating min and max as
  // we go:
  //
  T guess0 = guess;
  T multiplier = 2;
  T f_current = f0;

  if (std::abs(min) < std::abs(max)) {
    while (--count && ((f_current < 0) == (f0 < 0))) {
      max = guess;
      guess /= multiplier;
      if (guess < min) {
        guess = min;
        f_current = -f_current;  // There must be a change of sign!
        break;
      }
      multiplier *= 2;
      Unpack_0(f(guess), f_current);
    }
  } else {
    //
    // If min and max are negative we have to multiply to head towards min:
    //
    while (--count && ((f_current < 0) == (f0 < 0))) {
      max = guess;
      guess *= multiplier;
      if (guess < min) {
        guess = min;
        f_current = -f_current;  // There must be a change of sign!
        break;
      }
      multiplier *= 2;
      Unpack_0(f(guess), f_current);
    }
  }

  if (count) {
    min = guess;
    if (multiplier > 16)
      return (guess0 - guess) +
             BracketRootTowardsMax(f, guess, f_current, min, max, count);
  }
  return guess0 - (max + min) / 2;
}

template <class T>
T FloatDistance(const T& a, const T& b) {
  static_assert(std::numeric_limits<T>::is_specialized);
  // static_assert(std::numeric_limits<T>::radix != 2);

  //
  // Error handling:
  //
  if (!(std::isfinite(a))) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Utils::FloatDistance]" << std::endl
        << "[roots]: Argument a must be finite" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (!(std::isfinite(b))) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Utils::FloatDistance]" << std::endl
        << "[roots]: Argument b must be finite" << std::endl;
    ATHENA_ERROR(msg);
  }
  //
  // Special cases:
  //
  if (a > b) return -FloatDistance(b, a);
  if (a == b) return T(0);
  if (a == 0)
    return 1 +
           std::abs(FloatDistance(
               static_cast<T>((b < 0) ? T(-std::numeric_limits<T>::denorm_min())
                                      : std::numeric_limits<T>::denorm_min()),
               b));
  if (b == 0)
    return 1 +
           std::abs(FloatDistance(
               static_cast<T>((a < 0) ? T(-std::numeric_limits<T>::denorm_min())
                                      : std::numeric_limits<T>::denorm_min()),
               a));
  if (SIGN(a) != SIGN(b))
    return 2 +
           std::abs(FloatDistance(
               static_cast<T>((b < 0) ? T(-std::numeric_limits<T>::denorm_min())
                                      : std::numeric_limits<T>::denorm_min()),
               b)) +
           std::abs(FloatDistance(
               static_cast<T>((a < 0) ? T(-std::numeric_limits<T>::denorm_min())
                                      : std::numeric_limits<T>::denorm_min()),
               a));
  //
  // By the time we get here, both a and b must have the same sign, we want
  // b > a and both postive for the following logic:
  //
  if (a < 0) return FloatDistance(static_cast<T>(-b), static_cast<T>(-a));
  assert(a >= 0);
  assert(b >= a);

  std::int64_t expon;
  //
  // Note that if a is a denorm then the usual formula fails
  // because we actually have fewer than tools::digits<T>()
  // significant bits in the representation:
  //
  expon = 1 + std::ilogb(((std::fpclassify)(a) == (int)FP_SUBNORMAL)
                             ? (std::numeric_limits<T>::min)()
                             : a);
  T upper = std::scalbn(T(1), expon);
  T result = T(0);
  //
  // If b is greater than upper, then we *must* split the calculation
  // as the size of the ULP changes with each order of magnitude change:
  //
  if (b > upper) {
    std::int64_t expon2 = 1 + std::ilogb(b);
    T upper2 = std::scalbn(T(1), expon2 - 1);
    result = FloatDistance(upper2, b);
    result += (expon2 - expon - 1) *
              std::scalbn(T(1), std::numeric_limits<T>::digits - 1);
  }
  //
  // Use compensated double-double addition to avoid rounding
  // errors in the subtraction:
  //
  expon = std::numeric_limits<T>::digits - expon;
  T mb, x, y, z;
  if (((std::fpclassify)(a) == (int)FP_SUBNORMAL) ||
      (b - a < (std::numeric_limits<T>::min)())) {
    //
    // Special case - either one end of the range is a denormal, or else the
    // difference is. The regular code will fail if we're using the SSE2
    // registers on Intel and either the FTZ or DAZ flags are set.
    //
    T a2 = std::scalbn(a, std::numeric_limits<T>::digits);
    T b2 = std::scalbn(b, std::numeric_limits<T>::digits);
    mb = -(std::min)(T(std::scalbn(upper, std::numeric_limits<T>::digits)), b2);
    x = a2 + mb;
    z = x - a2;
    y = (a2 - (x - z)) + (mb - z);

    expon -= std::numeric_limits<T>::digits;
  } else {
    mb = -(std::min)(upper, b);
    x = a + mb;
    z = x - a;
    y = (a - (x - z)) + (mb - z);
  }
  if (x < 0) {
    x = -x;
    y = -y;
  }
  result += std::scalbn(x, expon) + std::scalbn(y, expon);
  //
  // Result must be an integer:
  //
  assert(result == std::floor(result));
  return result;
}

template <class Stepper, class F, class T>
T SecondOrderRootFinder(
    F f, T guess, T min, T max, int digits,
    std::uint64_t& max_iter) noexcept(std::is_floating_point<T>::
                                          value&& noexcept(std::declval<F>()(
                                              std::declval<T>()))) {
  if (min >= max) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Utils::SecondOrderRootFinder]"
        << std::endl
        << "[roots]: Range arguments in wrong order" << std::endl;
    ATHENA_ERROR(msg);
  }

  T f0(0), f1, f2;
  T result = guess;

  T factor = std::ldexp(static_cast<T>(1.0), 1 - digits);
  T delta =
      (std::max)(T(10000000 * guess), T(10000000));  // arbitrarily large delta
  T last_f0 = 0;
  T delta1 = delta;
  T delta2 = delta;
  bool out_of_bounds_sentry = false;

  //
  // We use these to sanity check that we do actually bracket a root,
  // we update these to the function value when we update the endpoints
  // of the range.  Then, provided at some point we update both endpoints
  // checking that max_range_f * min_range_f <= 0 verifies there is a root
  // to be found somewhere.  Note that if there is no root, and we approach
  // a local minima, then the derivative will go to zero, and hence the next
  // step will jump out of bounds (or at least past the minima), so this
  // check *should* happen in pathological cases.
  //
  T max_range_f = 0;
  T min_range_f = 0;

  std::uint64_t count(max_iter);

  do {
    last_f0 = f0;
    delta2 = delta1;
    delta1 = delta;
    Detail::UnpackTuple(f(result), f0, f1, f2);
    --count;

    if (0 == f0) break;
    if (f1 == 0) {
      // Oops zero derivative!!!
      Detail::HandleZeroDerivative(f, last_f0, f0, delta, result, guess, min,
                                   max);
    } else {
      if (f2 != 0) {
        delta = Stepper::step(result, f0, f1, f2);
        if (delta * f1 / f0 < 0) {
          // Oh dear, we have a problem as Newton and Halley steps
          // disagree about which way we should move.  Probably
          // there is cancelation error in the calculation of the
          // Halley step, or else the derivatives are so small
          // that their values are basically trash.  We will move
          // in the direction indicated by a Newton step, but
          // by no more than twice the current guess value, otherwise
          // we can jump way out of bounds if we're not careful.
          // See https://svn.boost.org/trac/boost/ticket/8314.
          delta = f0 / f1;
          if (std::abs(delta) > 2 * std::abs(guess))
            delta = (delta < 0 ? -1 : 1) * 2 * std::abs(guess);
        }
      } else
        delta = f0 / f1;
    }

    T convergence = std::abs(delta / delta2);
    if ((convergence > 0.8) && (convergence < 2)) {
      // last two steps haven't converged.
      delta = (delta > 0) ? (result - min) / 2 : (result - max) / 2;
      if ((result != 0) && (std::abs(delta) > result))
        delta = SIGN(delta) * std::abs(result) *
                0.9f;  // protect against huge jumps!
      // reset delta2 so that this branch will *not* be taken on the
      // next iteration:
      delta2 = delta * 3;
      delta1 = delta * 3;
    }
    guess = result;
    result -= delta;

    // check for out of bounds step:
    if (result < min) {
      T diff =
          ((std::abs(min) < 1) && (std::abs(result) > 1) &&
           ((std::numeric_limits<T>::max)() / std::abs(result) < std::abs(min)))
              ? T(1000)
          : (std::abs(min) < 1) && (std::abs((std::numeric_limits<T>::max)() *
                                             min) < std::abs(result))
              ? ((min < 0) != (result < 0)) ? -(std::numeric_limits<T>::max)()
                                            : (std::numeric_limits<T>::max)()
              : T(result / min);
      if (std::abs(diff) < 1) diff = 1 / diff;
      if (!out_of_bounds_sentry && (diff > 0) && (diff < 3)) {
        // Only a small out of bounds step, lets assume that the result
        // is probably approximately at min:
        delta = 0.99f * (guess - min);
        result = guess - delta;
        out_of_bounds_sentry = true;  // only take this branch once!
      } else {
        if (std::abs(FloatDistance(min, max)) < 2) {
          result = guess = (min + max) / 2;
          break;
        }
        delta = BracketRootTowardsMin(f, guess, f0, min, max, count);
        result = guess - delta;
        guess = min;
        continue;
      }
    } else if (result > max) {
      T diff =
          ((std::abs(max) < 1) && (std::abs(result) > 1) &&
           ((std::numeric_limits<T>::max)() / std::abs(result) < std::abs(max)))
              ? T(1000)
              : T(result / max);
      if (std::abs(diff) < 1) diff = 1 / diff;
      if (!out_of_bounds_sentry && (diff > 0) && (diff < 3)) {
        // Only a small out of bounds step, lets assume that the result
        // is probably approximately at min:
        delta = 0.99f * (guess - max);
        result = guess - delta;
        out_of_bounds_sentry = true;  // only take this branch once!
      } else {
        if (std::abs(FloatDistance(min, max)) < 2) {
          result = guess = (min + max) / 2;
          break;
        }
        delta = BracketRootTowardsMax(f, guess, f0, min, max, count);
        result = guess - delta;
        guess = min;
        continue;
      }
    }
    // update brackets:
    if (delta > 0) {
      max = guess;
      max_range_f = f0;
    } else {
      min = guess;
      min_range_f = f0;
    }
    //
    // Sanity check that we bracket the root:
    //
    if (max_range_f * min_range_f > 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [Utils::SecondOrderRootFinder]"
          << std::endl
          << "[roots]: There appears to be no root to be found" << std::endl;
      ATHENA_ERROR(msg);
    }
  } while (count && (std::abs(result * factor) < std::abs(delta)));

  max_iter -= count;

  return result;
}
}  // namespace Detail

template <class F, class T>
T HalleyIterate(
    F f, T guess, T min, T max, int digits,
    std::uint64_t& max_iter) noexcept(std::is_floating_point<T>::
                                          value&& noexcept(std::declval<F>()(
                                              std::declval<T>()))) {
  return Detail::SecondOrderRootFinder<Detail::HalleyStep>(f, guess, min, max,
                                                           digits, max_iter);
}

template <class F, class T>
inline T HalleyIterate(F f, T guess, T min, T max, int digits) noexcept(
    std::is_floating_point<T>::value&& noexcept(
        std::declval<F>()(std::declval<T>()))) {
  std::uint64_t m = (std::numeric_limits<std::uint64_t>::max)();
  return HalleyIterate(f, guess, min, max, digits, m);
}

namespace Detail {

struct SchroderStepper {
  template <class T>
  static T step(const T& x, const T& f0, const T& f1,
                const T& f2) noexcept(std::is_floating_point<T>::value) {
    T ratio = f0 / f1;
    T delta;
    if ((x != 0) && (std::abs(ratio / x) < 0.1)) {
      delta = ratio + (f2 / (2 * f1)) * ratio * ratio;
      // check second derivative doesn't over compensate:
      if (delta * ratio < 0) delta = ratio;
    } else
      delta = ratio;  // fall back to Newton iteration.
    return delta;
  }
};

}  // namespace Detail

template <class F, class T>
T SchroderIterate(
    F f, T guess, T min, T max, int digits,
    std::uint64_t& max_iter) noexcept(std::is_floating_point<T>::
                                          value&& noexcept(std::declval<F>()(
                                              std::declval<T>()))) {
  return Detail::SecondOrderRootFinder<Detail::SchroderStepper>(
      f, guess, min, max, digits, max_iter);
}

template <class F, class T>
inline T SchroderIterate(F f, T guess, T min, T max, int digits) noexcept(
    std::is_floating_point<T>::value&& noexcept(
        std::declval<F>()(std::declval<T>()))) {
  std::uint64_t m = (std::numeric_limits<std::uint64_t>::max)();
  return SchroderIterate(f, guess, min, max, digits, m);
}
}  // namespace Utils

#endif  // UTILS_ROOTS_HPP_