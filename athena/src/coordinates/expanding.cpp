//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//======================================================================================
//! \file ExpandingCoord.cpp
//  \brief implements functions for expanding coordinates (with a1, a2, a3) in a
//  derived class of the Coordinates abstract base class.

// C headers

// C++ headers
#include <cmath>  // std::pow

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "coordinates.hpp"

//----------------------------------------------------------------------------------------
// Expanding coordinates constructor
ExpandingCoord::ExpandingCoord(MeshBlock *pmb, ParameterInput *pin, bool flag)
    : Coordinates(pmb, pin, flag),
      dot_a1_(pin->GetOrAddReal("coord", "dot_a1", 0.0)),
      dot_a2_(pin->GetOrAddReal("coord", "dot_a2", 0.0)),
      dot_a3_(pin->GetOrAddReal("coord", "dot_a3", 0.0)),
      a1_(1.0),
      a2_(1.0),
      a3_(1.0),
      a1_ratio_(1.0),
      a2_ratio_(1.0),
      a3_ratio_(1.0),
      a1_reciprocal_(1.0),
      a2_reciprocal_(1.0),
      a3_reciprocal_(1.0),
      a1_next_(1.0),
      a2_next_(1.0),
      a3_next_(1.0),
      a1_a2_(1.0),
      a1_a3_(1.0),
      a2_a3_(1.0),
      a1_a2_a3_(1.0) {
  // initialize volume - averaged coordinates and spacing
  // x1-direction: x1v = dx/2
  for (int i = il - ng; i <= iu + ng; ++i) {
    x1v(i) = 0.5 * (x1f(i + 1) + x1f(i));
  }
  for (int i = il - ng; i <= iu + ng - 1; ++i) {
    if (pmb->block_size.x1rat != 1.0) {
      dx1v(i) = x1v(i + 1) - x1v(i);
    } else {
      // dx1v = dx1f constant for uniform mesh; may disagree with x1v(i+1) -
      // x1v(i)
      dx1v(i) = dx1f(i);
    }
  }

  // x2-direction: x2v = dy/2
  if (pmb->block_size.nx2 == 1) {
    x2v(jl) = 0.5 * (x2f(jl + 1) + x2f(jl));
    dx2v(jl) = dx2f(jl);
  } else {
    for (int j = jl - ng; j <= ju + ng; ++j) {
      x2v(j) = 0.5 * (x2f(j + 1) + x2f(j));
    }
    for (int j = jl - ng; j <= ju + ng - 1; ++j) {
      if (pmb->block_size.x2rat != 1.0) {
        dx2v(j) = x2v(j + 1) - x2v(j);
      } else {
        // dx2v = dx2f constant for uniform mesh; may disagree with x2v(j+1) -
        // x2v(j)
        dx2v(j) = dx2f(j);
      }
    }
  }

  // x3-direction: x3v = dz/2
  if (pmb->block_size.nx3 == 1) {
    x3v(kl) = 0.5 * (x3f(kl + 1) + x3f(kl));
    dx3v(kl) = dx3f(kl);
  } else {
    for (int k = kl - ng; k <= ku + ng; ++k) {
      x3v(k) = 0.5 * (x3f(k + 1) + x3f(k));
    }
    for (int k = kl - ng; k <= ku + ng - 1; ++k) {
      if (pmb->block_size.x3rat != 1.0) {
        dx3v(k) = x3v(k + 1) - x3v(k);
      } else {
        // dxkv = dx3f constant for uniform mesh; may disagree with x3v(k+1) -
        // x3v(k)
        dx3v(k) = dx3f(k);
      }
    }
  }
  // initialize geometry coefficients
  // TODO(sxc18): compute the exact geometry coefficient in the expanding
  // coordinates x1-direction
  for (int i = il - ng; i <= iu + ng; ++i) {
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
    for (int j = jl - ng; j <= ju + ng; ++j) {
      h32v(j) = 1.0;
      h32f(j) = 1.0;
      dh32vd2(j) = 0.0;
      dh32fd2(j) = 0.0;
    }
  }

  // initialize area-averaged coordinates used with MHD AMR
  if ((pmb->pmy_mesh->multilevel) && MAGNETIC_FIELDS_ENABLED) {
    for (int i = il - ng; i <= iu + ng; ++i) {
      x1s2(i) = x1s3(i) = x1v(i);
    }
    if (pmb->block_size.nx2 == 1) {
      x2s1(jl) = x2s3(jl) = x2v(jl);
    } else {
      for (int j = jl - ng; j <= ju + ng; ++j) {
        x2s1(j) = x2s3(j) = x2v(j);
      }
    }
    if (pmb->block_size.nx3 == 1) {
      x3s1(kl) = x3s2(kl) = x3v(kl);
    } else {
      for (int k = kl - ng; k <= ku + ng; ++k) {
        x3s1(k) = x3s2(k) = x3v(k);
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void ExpandingCoord::UpdateCoords(const Real &time, const Real &dt)
//  \brief update coordinate factors with time.

void ExpandingCoord::UpdateCoords(const Real &time, const Real &dt) {
  const Real time_next(time + dt);
  //if (dot_a1_ > 0) {
  //  a1_ = 1.0 + time * dot_a1_;
  //  a1_next_ = 1.0 + time_next * dot_a1_;
  //} else {
  //  a1_ = 1.0 / (1.0 - time * dot_a1_);
  //  a1_next_ = 1.0 / (1.0 - time_next * dot_a1_);
  //}
  //if (dot_a2_ > 0) {
  //  a2_ = 1.0 + time * dot_a2_;
  //  a2_next_ = 1.0 + time_next * dot_a2_;
  //} else {
  //  a2_ = 1.0 / (1.0 - time * dot_a2_);
  //  a2_next_ = 1.0 / (1.0 - time_next * dot_a2_);
  //}
  //if (dot_a3_ > 0) {
  //  a3_ = 1.0 + time * dot_a3_;
  //  a3_next_ = 1.0 + time_next * dot_a3_;
  //} else {
  //  a3_ = 1.0 / (1.0 - time * dot_a3_);
  //  a3_next_ = 1.0 / (1.0 - time_next * dot_a3_);
  //}
  if (dot_a2_ > 0) {
    a2_ = 1.0 + time * dot_a2_;
    a2_next_ = 1.0 + time_next * dot_a2_;
  } else {
    a2_ = 1.0 / (1.0 - time * dot_a2_);
    a2_next_ = 1.0 / (1.0 - time_next * dot_a2_);
  }
  a3_ = a2_;
  a3_next_ = a2_next_;
  a1_ = SQR(a2_);
  a1_next_ = SQR(a2_next_);

  a1_reciprocal_ = 1.0 / a1_;
  a2_reciprocal_ = 1.0 / a2_;
  a3_reciprocal_ = 1.0 / a3_;
  a1_a2_ = a1_ * a2_;
  a1_a3_ = a1_ * a3_;
  a2_a3_ = a2_ * a3_;
  a1_a2_a3_ = a1_ * a2_ * a3_;
  a1_ratio_ = a1_ / a1_next_;
  a2_ratio_ = a2_ / a2_next_;
  a3_ratio_ = a3_ / a3_next_;
}

//----------------------------------------------------------------------------------------
// EdgeXLength functions: compute physical length at cell edge-X as vector
// Edge1(i,j,k) located at (i,j-1/2,k-1/2), i.e. (x1v(i), x2f(j), x3f(k))

void ExpandingCoord::Edge1Length(const int k, const int j, const int il,
                                 const int iu, AthenaArray<Real> &len) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    len(i) = dx1f(i) * a1_;
  }
  return;
}

// Edge2(i,j,k) located at (i-1/2,j,k-1/2), i.e. (x1f(i), x2v(j), x3f(k))

void ExpandingCoord::Edge2Length(const int k, const int j, const int il,
                                 const int iu, AthenaArray<Real> &len) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    len(i) = dx2f(j) * a2_;
  }
  return;
}

// Edge3(i,j,k) located at (i-1/2,j-1/2,k), i.e. (x1f(i), x2f(j), x3v(k))

void ExpandingCoord::Edge3Length(const int k, const int j, const int il,
                                 const int iu, AthenaArray<Real> &len) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    len(i) = dx3f(k) * a3_;
  }
  return;
}

//----------------------------------------------------------------------------------------
// GetEdgeXLength functions: return length of edge-X at (i,j,k)

Real ExpandingCoord::GetEdge1Length(const int k, const int j, const int i) {
  return dx1f(i) * a1_;
}

Real ExpandingCoord::GetEdge2Length(const int k, const int j, const int i) {
  return dx2f(j) * a2_;
}

Real ExpandingCoord::GetEdge3Length(const int k, const int j, const int i) {
  return dx3f(k) * a3_;
}

//----------------------------------------------------------------------------------------
// VolCenterXLength functions: compute physical length connecting cell centers
// as vector VolCenter1(i,j,k) located at (i+1/2,j,k), i.e. (x1f(i+1), x2v(j),
// x3v(k))
void ExpandingCoord::VolCenter1Length(const int k, const int j, const int il,
                                      const int iu, AthenaArray<Real> &len) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    len(i) = dx1v(i) * a1_;
  }
  return;
}
void ExpandingCoord::VolCenter2Length(const int k, const int j, const int il,
                                      const int iu, AthenaArray<Real> &len) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    len(i) = dx2v(j) * a2_;
  }
  return;
}
void ExpandingCoord::VolCenter3Length(const int k, const int j, const int il,
                                      const int iu, AthenaArray<Real> &len) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    len(i) = dx3v(k) * a3_;
  }
  return;
}

//----------------------------------------------------------------------------------------
// CenterWidthX functions: return physical width in X-dir at (i,j,k) cell-center

void ExpandingCoord::CenterWidth1(const int k, const int j, const int il,
                                  const int iu, AthenaArray<Real> &dx1) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    dx1(i) = dx1f(i) * a1_;
  }
  return;
}

void ExpandingCoord::CenterWidth2(const int k, const int j, const int il,
                                  const int iu, AthenaArray<Real> &dx2) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    dx2(i) = dx2f(j) * a2_;
  }
  return;
}

void ExpandingCoord::CenterWidth3(const int k, const int j, const int il,
                                  const int iu, AthenaArray<Real> &dx3) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    dx3(i) = dx3f(k) * a3_;
  }
  return;
}

//----------------------------------------------------------------------------------------
// FaceXArea functions: compute area of face with normal in X-dir as vector

void ExpandingCoord::Face1Area(const int k, const int j, const int il,
                               const int iu, AthenaArray<Real> &area) {
#pragma nounroll
  for (int i = il; i <= iu; ++i) {
    // area1 = dy dz
    Real &area_i = area(i);
    area_i = dx2f(j) * dx3f(k) * a2_a3_;
  }
  return;
}

void ExpandingCoord::Face2Area(const int k, const int j, const int il,
                               const int iu, AthenaArray<Real> &area) {
#pragma nounroll
  for (int i = il; i <= iu; ++i) {
    // area2 = dx dz
    Real &area_i = area(i);
    area_i = dx1f(i) * dx3f(k) * a1_a3_;
  }
  return;
}

void ExpandingCoord::Face3Area(const int k, const int j, const int il,
                               const int iu, AthenaArray<Real> &area) {
#pragma nounroll
  for (int i = il; i <= iu; ++i) {
    // area3 = dx dy
    Real &area_i = area(i);
    area_i = dx1f(i) * dx2f(j) * a1_a2_;
  }
  return;
}

//----------------------------------------------------------------------------------------
// GetFaceXArea functions: return area of face with normal in X-dir at (i,j,k)

Real ExpandingCoord::GetFace1Area(const int k, const int j, const int i) {
  return dx2f(j) * dx3f(k) * a2_a3_;
}

Real ExpandingCoord::GetFace2Area(const int k, const int j, const int i) {
  return dx1f(i) * dx3f(k) * a1_a3_;
}

Real ExpandingCoord::GetFace3Area(const int k, const int j, const int i) {
  return dx1f(i) * dx2f(j) * a1_a2_;
}

//----------------------------------------------------------------------------------------
// VolCenterFaceXArea functions: compute area of face with normal in X-dir as
// vector where the faces are joined by cell centers (for non-ideal MHD)

void ExpandingCoord::VolCenterFace1Area(const int k, const int j, const int il,
                                        const int iu, AthenaArray<Real> &area) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    Real &area_i = area(i);
    area_i = dx2v(j) * dx3v(k) * a2_a3_;
  }
  return;
}

void ExpandingCoord::VolCenterFace2Area(const int k, const int j, const int il,
                                        const int iu, AthenaArray<Real> &area) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    Real &area_i = area(i);
    area_i = dx1v(i) * dx3v(k) * a1_a3_;
  }
  return;
}

void ExpandingCoord::VolCenterFace3Area(const int k, const int j, const int il,
                                        const int iu, AthenaArray<Real> &area) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    Real &area_i = area(i);
    area_i = dx1v(i) * dx2v(j) * a1_a2_;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Cell Volume function: compute volume of cell as vector

void ExpandingCoord::CellVolume(const int k, const int j, const int il,
                                const int iu, AthenaArray<Real> &vol) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    // volume = dx dy dz
    Real &vol_i = vol(i);
    vol_i = dx1f(i) * dx2f(j) * dx3f(k) * a1_a2_a3_;
  }
  return;
}

//----------------------------------------------------------------------------------------
// GetCellVolume: returns cell volume at (i,j,k)

Real ExpandingCoord::GetCellVolume(const int k, const int j, const int i) {
  return dx1f(i) * dx2f(j) * dx3f(k) * a1_a2_a3_;
}

//----------------------------------------------------------------------------------------
// Coordinate (Geometric) source term function
void ExpandingCoord::AddCoordTermsDivergence(const CoordTermsDivergence &flag,
                                             const Real dt,
                                             const AthenaArray<Real> *flux,
                                             const AthenaArray<Real> &prim,
                                             const AthenaArray<Real> &bcc,
                                             AthenaArray<Real> &u) {
  if (flag == CoordTermsDivergence::stationary) return;

  int il = pmy_block->is, iu = pmy_block->ie, jl = pmy_block->js,
      ju = pmy_block->je, kl = pmy_block->ks, ku = pmy_block->ke;
  if (pmy_block->pbval->nblevel[1][1][0] != -1) il -= NGHOST;
  if (pmy_block->pbval->nblevel[1][1][2] != -1) iu += NGHOST;
  if (pmy_block->pbval->nblevel[1][0][1] != -1) jl -= NGHOST;
  if (pmy_block->pbval->nblevel[1][2][1] != -1) ju += NGHOST;
  if (pmy_block->pbval->nblevel[0][1][1] != -1) kl -= NGHOST;
  if (pmy_block->pbval->nblevel[2][1][1] != -1) ku += NGHOST;

  if (flag == CoordTermsDivergence::preliminary) {
    const Real gm(pmy_block->peos->GetGamma());
    const Real tmp_den(a1_ratio_ * a2_ratio_ * a3_ratio_);
    const Real tmp_mom1(tmp_den * a1_ratio_), tmp_mom2(tmp_den * a2_ratio_),
        tmp_mom3(tmp_den * a3_ratio_);
    const Real tmp_energy(std::pow(tmp_den, gm));
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
#pragma omp simd
        for (int i = il; i <= iu; ++i) {
          // src_1 in energy equation: must be done at the begining
          if (NON_BAROTROPIC_EOS) {
            Real pb(0);
            const Real u_m1(u(IM1, k, j, i));
            const Real u_m2(u(IM2, k, j, i));
            const Real u_m3(u(IM3, k, j, i));
            const Real di((1.0 / u(IDN, k, j, i)));
            const Real e_k(0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3)));
            if (MAGNETIC_FIELDS_ENABLED) {
              const Real bcc1(bcc(IB1, k, j, i));
              const Real bcc2(bcc(IB2, k, j, i));
              const Real bcc3(bcc(IB3, k, j, i));
              pb = 0.5 * (SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
            }
            u(IEN, k, j, i) -= e_k + pb;  // internal energy
            u(IEN, k, j, i) = tmp_energy * u(IEN, k, j, i);
          }

          // src_2 in density equation: - \rho * \frac{\dot{l}}{l}
          u(IDN, k, j, i) = tmp_den * u(IDN, k, j, i);

          // src_3 in momemntum equation: - \rho * v^i (\frac{\dot{l}}{l} +
          // \frac{\dot{a^i}}{a^i})
          u(IM1, k, j, i) = tmp_mom1 * u(IM1, k, j, i);
          u(IM2, k, j, i) = tmp_mom2 * u(IM2, k, j, i);
          u(IM3, k, j, i) = tmp_mom3 * u(IM3, k, j, i);
        }
      }
    }
  }

  if (flag == CoordTermsDivergence::finale) {
    if (NON_BAROTROPIC_EOS) {
      for (int k = kl; k <= ku; ++k) {
        for (int j = jl; j <= ju; ++j) {
#pragma omp simd
          for (int i = il; i <= iu; ++i) {
            Real pb(0);
            const Real u_m1(u(IM1, k, j, i));
            const Real u_m2(u(IM2, k, j, i));
            const Real u_m3(u(IM3, k, j, i));
            const Real di((1.0 / u(IDN, k, j, i)));
            const Real e_k(0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3)));
            if (MAGNETIC_FIELDS_ENABLED) {
              const Real bcc1(bcc(IB1, k, j, i));
              const Real bcc2(bcc(IB2, k, j, i));
              const Real bcc3(bcc(IB3, k, j, i));
              pb = 0.5 * (SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
            }
            u(IEN, k, j, i) += e_k + pb;
          }
        }
      }
    } else
      pmy_block->peos->UpdateIsoSoundSpeed(
          pmy_block->peos->GetIsoSoundSpeed() *
          std::pow(a1_ * a2_ * a3_ / (a1_next_ * a2_next_ * a3_next_),
                   1.0 / 3.0));
  }

  return;
}

//----------------------------------------------------------------------------------------
// Coordinate (Geometric) source term function for CT of field
// This method statifies CT, with updating through ratio of area
void ExpandingCoord::AddCoordTermsCT(const CoordTermsDivergence &flag,
                                     const Real dt,
                                     const AthenaArray<Real> &bcc,
                                     FaceField &b_out) {
  if (flag == CoordTermsDivergence::stationary) return;
  MeshBlock *pmb = pmy_block;
  int il = pmy_block->is, iu = pmy_block->ie, jl = pmy_block->js,
      ju = pmy_block->je, kl = pmy_block->ks, ku = pmy_block->ke;
  if (pmy_block->pbval->nblevel[1][1][0] != -1) il -= NGHOST;
  if (pmy_block->pbval->nblevel[1][1][2] != -1) iu += NGHOST;
  if (pmy_block->pbval->nblevel[1][0][1] != -1) jl -= NGHOST;
  if (pmy_block->pbval->nblevel[1][2][1] != -1) ju += NGHOST;
  if (pmy_block->pbval->nblevel[0][1][1] != -1) kl -= NGHOST;
  if (pmy_block->pbval->nblevel[2][1][1] != -1) ku += NGHOST;
  Real tmp(0);

  //---- update B1
  tmp = a2_ratio_ * a3_ratio_;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
#pragma omp simd
      for (int i = il; i <= iu + 1; ++i) {
        b_out.x1f(k, j, i) = b_out.x1f(k, j, i) * tmp;
      }
    }
  }

  //---- update B2
  tmp = a1_ratio_ * a3_ratio_;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju + 1; ++j) {
#pragma omp simd
      for (int i = il; i <= iu; ++i) {
        b_out.x2f(k, j, i) = b_out.x2f(k, j, i) * tmp;
      }
    }
  }

  //---- update B3
  tmp = a1_ratio_ * a2_ratio_;
  for (int k = kl; k <= ku + 1; ++k) {
    for (int j = jl; j <= ju; ++j) {
#pragma omp simd
      for (int i = il; i <= iu; ++i) {
        b_out.x3f(k, j, i) = b_out.x3f(k, j, i) * tmp;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Coordinate (Geometric) source term function for particle
// This method is a kind of operator split

void ExpandingCoord::AddCoordTermsPar(const CoordTermsDivergence &flag,
                                      const Real &time, const Real dt,
                                      const AthenaArray<Real> &meshsrc,
                                      const int &npar, AthenaArray<Real> &v1,
                                      AthenaArray<Real> &v2,
                                      AthenaArray<Real> &v3) {
  if (flag == CoordTermsDivergence::stationary) return;
  Real a1_ratio_half, a2_ratio_half, a3_ratio_half;
  /*if (dot_a1_ > 0)
    a1_ratio_half = (1.0 + time * dot_a1_) / (1.0 + (time + dt) * dot_a1_);
  else
    a1_ratio_half = (1.0 - (time + dt) * dot_a1_) / (1.0 - time * dot_a1_);
  if (dot_a2_ > 0)
    a2_ratio_half = (1.0 + time * dot_a2_) / (1.0 + (time + dt) * dot_a2_);
  else
    a2_ratio_half = (1.0 - (time + dt) * dot_a2_) / (1.0 - time * dot_a2_);
  if (dot_a3_ > 0)
    a3_ratio_half = (1.0 + time * dot_a3_) / (1.0 + (time + dt) * dot_a3_);
  else
    a3_ratio_half = (1.0 - (time + dt) * dot_a3_) / (1.0 - time * dot_a3_);*/

  if (dot_a2_ > 0)
    a2_ratio_half = (1.0 + time * dot_a2_) / (1.0 + (time + dt) * dot_a2_);
  else
    a2_ratio_half = (1.0 - (time + dt) * dot_a2_) / (1.0 - time * dot_a2_);
  a3_ratio_half = a2_ratio_half;
  a1_ratio_half = SQR(a2_ratio_half);
  
#pragma omp simd simdlen(SIMD_WIDTH)
  for (int k = 0; k < npar; ++k) {
    v1(k) *= a1_ratio_half;
    v2(k) *= a2_ratio_half;
    v3(k) *= a3_ratio_half;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void ExpandingCoord::CartesianToMeshCoords(const Real &x, const Real &y,
//                                           const Real &z, Real &x1, Real &x2,
//                                           Real &x3) const
//  \brief returns in (x1, x2, x3) the coordinates used by the mesh from
//  Cartesian coordinates (x, y, z).

void ExpandingCoord::CartesianToMeshCoords(const Real &x, const Real &y,
                                           const Real &z, Real &x1, Real &x2,
                                           Real &x3) const {
  x1 = x * a1_reciprocal_;
  x2 = y * a2_reciprocal_;
  x3 = z * a3_reciprocal_;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void ExpandingCoord::MeshCoordsToCartesian(const Real &x1, const Real
//! &x2,
//                                           const Real &x3, Real &x, Real &y,
//                                           Real &z) const
//  \brief returns in Cartesian coordinates (x, y, z) from (x1, x2, x3) the
//  coordinates used by the mesh.

void ExpandingCoord::MeshCoordsToCartesian(const Real &x1, const Real &x2,
                                           const Real &x3, Real &x, Real &y,
                                           Real &z) const {
  x = x1 * a1_;
  y = x2 * a2_;
  z = x3 * a3_;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void ExpandingCoord::CartesianToMeshCoordsVector(const Real &x, const
//! Real &y,
//                                                 const Real &z, const Real
//                                                 &vx, const Real &vy, const
//                                                 Real &vz, Real &vx1, Real
//                                                 &vx2,
//                                                Real &vx3) const
//  \brief returns in (vx1, vx2, vx3) the components of a vector in Mesh
//  coordinates when the vector is (vx, vy, vz) at (x, y, z) in Cartesian
//  coordinates.

void ExpandingCoord::CartesianToMeshCoordsVector(const Real &x, const Real &y,
                                                 const Real &z, const Real &vx,
                                                 const Real &vy, const Real &vz,
                                                 Real &vx1, Real &vx2,
                                                 Real &vx3) const {
  vx1 = vx;
  vx2 = vy;
  vx3 = vz;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void ExpandingCoord::MeshCoordsToCartesianVector(
//    const Real &x1, const Real &x2, const Real &x3, const Real &vx1,
//    const Real &vx2, const Real &vx3, Real &vx, Real &vy, Real &vz) const
//  \brief returns in (vx, vy, vz) the components of a vector in Cartesian
//  coordinates when the vector is (vx1, vy1, vz1) at (x1, x2, x3) in Mesh
//  coordinates.

void ExpandingCoord::MeshCoordsToCartesianVector(
    const Real &x1, const Real &x2, const Real &x3, const Real &vx1,
    const Real &vx2, const Real &vx3, Real &vx, Real &vy, Real &vz) const {
  vx = vx1;
  vy = vx2;
  vz = vx3;
  return;
}
