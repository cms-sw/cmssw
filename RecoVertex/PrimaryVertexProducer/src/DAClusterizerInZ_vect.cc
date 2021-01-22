#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZ_vect.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

#include <cmath>
#include <cassert>
#include <limits>
#include <iomanip>
#include "FWCore/Utilities/interface/isFinite.h"
#include "vdt/vdtMath.h"

#include <Math/SMatrix.h>

using namespace std;

//#define DEBUG
#ifdef DEBUG
#define DEBUGLEVEL 0
#endif

DAClusterizerInZ_vect::DAClusterizerInZ_vect(const edm::ParameterSet& conf) {
  // hardcoded parameters
  maxIterations_ = 1000;
  mintrkweight_ = 0.5;

  // configurable debug outptut debug output
  verbose_ = conf.getUntrackedParameter<bool>("verbose", false);
  zdumpcenter_ = conf.getUntrackedParameter<double>("zdumpcenter", 0.);
  zdumpwidth_ = conf.getUntrackedParameter<double>("zdumpwidth", 20.);

  // configurable parameters
  double Tmin = conf.getParameter<double>("Tmin");
  double Tpurge = conf.getParameter<double>("Tpurge");
  double Tstop = conf.getParameter<double>("Tstop");
  vertexSize_ = conf.getParameter<double>("vertexSize");
  coolingFactor_ = conf.getParameter<double>("coolingFactor");
  d0CutOff_ = conf.getParameter<double>("d0CutOff");
  dzCutOff_ = conf.getParameter<double>("dzCutOff");
  uniquetrkweight_ = conf.getParameter<double>("uniquetrkweight");
  zmerge_ = conf.getParameter<double>("zmerge");
  sel_zrange_ = conf.getParameter<double>("zrange");
  convergence_mode_ = conf.getParameter<int>("convergence_mode");
  delta_lowT_ = conf.getParameter<double>("delta_lowT");
  delta_highT_ = conf.getParameter<double>("delta_highT");

  if (verbose_) {
    std::cout << "DAClusterizerinZ_vect: mintrkweight = " << mintrkweight_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: uniquetrkweight = " << uniquetrkweight_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: zmerge = " << zmerge_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: Tmin = " << Tmin << std::endl;
    std::cout << "DAClusterizerinZ_vect: Tpurge = " << Tpurge << std::endl;
    std::cout << "DAClusterizerinZ_vect: Tstop = " << Tstop << std::endl;
    std::cout << "DAClusterizerinZ_vect: vertexSize = " << vertexSize_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: coolingFactor = " << coolingFactor_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: d0CutOff = " << d0CutOff_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: dzCutOff = " << dzCutOff_ << std::endl;
    std::cout << "DAClusterizerInZ_vect: zrange = " << sel_zrange_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: convergence mode = " << convergence_mode_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: delta_highT = " << delta_highT_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: delta_lowT = " << delta_lowT_ << std::endl;
  }
#ifdef DEBUG
  std::cout << "DAClusterizerinZ_vect: DEBUGLEVEL " << DEBUGLEVEL << std::endl;
#endif

  if (convergence_mode_ > 1) {
    edm::LogWarning("DAClusterizerinZ_vect")
        << "DAClusterizerInZ_vect: invalid convergence_mode" << convergence_mode_ << "  reset to default " << 0;
    convergence_mode_ = 0;
  }

  if (Tmin == 0) {
    edm::LogWarning("DAClusterizerinZ_vect")
        << "DAClusterizerInZ_vect: invalid Tmin" << Tmin << "  reset to default " << 1. / betamax_;
  } else {
    betamax_ = 1. / Tmin;
  }

  if ((Tpurge > Tmin) || (Tpurge == 0)) {
    edm::LogWarning("DAClusterizerinZ_vect")
        << "DAClusterizerInZ_vect: invalid Tpurge" << Tpurge << "  set to " << Tmin;
    Tpurge = Tmin;
  }
  betapurge_ = 1. / Tpurge;

  if ((Tstop > Tpurge) || (Tstop == 0)) {
    edm::LogWarning("DAClusterizerinZ_vect")
        << "DAClusterizerInZ_vect: invalid Tstop" << Tstop << "  set to  " << max(1., Tpurge);
    Tstop = max(1., Tpurge);
  }
  betastop_ = 1. / Tstop;
}

namespace {
  inline double local_exp(double const& inp) { return vdt::fast_exp(inp); }

  inline void local_exp_list(double const* __restrict__ arg_inp, double* __restrict__ arg_out, const int arg_arr_size) {
    for (auto i = 0; i != arg_arr_size; ++i)
      arg_out[i] = vdt::fast_exp(arg_inp[i]);
  }

  inline void local_exp_list_range(double const* __restrict__ arg_inp,
                                   double* __restrict__ arg_out,
                                   const int kmin,
                                   const int kmax) {
    for (auto i = kmin; i != kmax; ++i)
      arg_out[i] = vdt::fast_exp(arg_inp[i]);
  }

}  // namespace

void DAClusterizerInZ_vect::verify(const vertex_t& v, const track_t& tks, unsigned int nv, unsigned int nt) const {
  if (!(nv == 999999)) {
    assert(nv == v.getSize());
  } else {
    nv = v.getSize();
  }

  if (!(nt == 999999)) {
    assert(nt == tks.getSize());
  } else {
    nt = tks.getSize();
  }

  assert(v.z.size() == nv);
  assert(v.pk.size() == nv);
  assert(v.swz.size() == nv);
  assert(v.ei_cache.size() == nv);
  assert(v.ei.size() == nv);
  assert(v.se.size() == nv);
  assert(v.swz.size() == nv);
  assert(v.swE.size() == nv);

  assert(v.z_ptr == &v.z.front());
  assert(v.pk_ptr == &v.pk.front());
  assert(v.ei_cache_ptr == &v.ei_cache.front());
  assert(v.swz_ptr == &v.swz.front());
  assert(v.se_ptr == &v.se.front());
  assert(v.swE_ptr == &v.swE.front());

  for (unsigned int k = 0; k < nv - 1; k++) {
    if (v.z[k] <= v.z[k + 1])
      continue;
    cout << " Z, cluster z-ordering assertion failure   z[" << k << "] =" << v.z[k] << "    z[" << k + 1
         << "] =" << v.z[k + 1] << endl;
  }
  //for(unsigned int k=0; k< nv-1; k++){
  //  assert( v.z[k] <= v.z[k+1]);
  //}

  assert(nt == tks.z.size());
  assert(nt == tks.dz2.size());
  assert(nt == tks.tt.size());
  assert(nt == tks.pi.size());
  assert(nt == tks.Z_sum.size());
  assert(nt == tks.kmin.size());
  assert(nt == tks.kmax.size());

  assert(tks.z_ptr == &tks.z.front());
  assert(tks.dz2_ptr == &tks.dz2.front());
  assert(tks.pi_ptr == &tks.pi.front());
  assert(tks.Z_sum_ptr == &tks.Z_sum.front());

  for (unsigned int i = 0; i < nt; i++) {
    if ((tks.kmin[i] < tks.kmax[i]) && (tks.kmax[i] <= nv))
      continue;
    cout << "track vertex range assertion failure" << i << "/" << nt << "   kmin,kmax=" << tks.kmin[i] << ", "
         << tks.kmax[i] << "  nv=" << nv << endl;
  }

  for (unsigned int i = 0; i < nt; i++) {
    assert((tks.kmin[i] < tks.kmax[i]) && (tks.kmax[i] <= nv));
  }
}

//todo: use r-value possibility of c++11 here
DAClusterizerInZ_vect::track_t DAClusterizerInZ_vect::fill(const vector<reco::TransientTrack>& tracks) const {
  // prepare track data for clustering
  track_t tks;
  for (auto it = tracks.begin(); it != tracks.end(); it++) {
    if (!(*it).isValid())
      continue;
    double t_pi = 1.;
    double t_z = ((*it).stateAtBeamLine().trackStateAtPCA()).position().z();
    if (std::fabs(t_z) > 1000.)
      continue;
    auto const& t_mom = (*it).stateAtBeamLine().trackStateAtPCA().momentum();
    //  get the beam-spot
    reco::BeamSpot beamspot = (it->stateAtBeamLine()).beamSpot();
    double t_dz2 = std::pow((*it).track().dzError(), 2)  // track errror
                   + (std::pow(beamspot.BeamWidthX() * t_mom.x(), 2) + std::pow(beamspot.BeamWidthY() * t_mom.y(), 2)) *
                         std::pow(t_mom.z(), 2) / std::pow(t_mom.perp2(), 2)  // beam spot width
                   + std::pow(vertexSize_, 2);  // intrinsic vertex size, safer for outliers and short lived decays
    t_dz2 = 1. / t_dz2;
    if (edm::isNotFinite(t_dz2) || t_dz2 < std::numeric_limits<double>::min())
      continue;
    if (d0CutOff_ > 0) {
      Measurement1D atIP = (*it).stateAtBeamLine().transverseImpactParameter();  // error contains beamspot
      t_pi = 1. / (1. + local_exp(std::pow(atIP.value() / atIP.error(), 2) -
                                  std::pow(d0CutOff_, 2)));  // reduce weight for high ip tracks
      if (edm::isNotFinite(t_pi) || t_pi < std::numeric_limits<double>::epsilon())
        continue;  // usually is > 0.99
    }
    LogTrace("DAClusterizerinZ_vect") << t_z << ' ' << t_dz2 << ' ' << t_pi;
    tks.addItemSorted(t_z, t_dz2, &(*it), t_pi);
  }

  tks.extractRaw();
#ifdef DEBUG
  if (DEBUGLEVEL > 0) {
    std::cout << "Track count (Z) " << tks.getSize() << std::endl;
  }
#endif

  return tks;
}

namespace {
  inline double Eik(double t_z, double k_z, double t_dz2) { return std::pow(t_z - k_z, 2) * t_dz2; }
}  // namespace

void DAClusterizerInZ_vect::set_vtx_range(double beta, track_t& gtracks, vertex_t& gvertices) const {
  const unsigned int nv = gvertices.getSize();
  const unsigned int nt = gtracks.getSize();

  if (nv == 0) {
    edm::LogWarning("DAClusterizerinZ_vect") << "empty cluster list in set_vtx_range";
    return;
  }

  for (auto itrack = 0U; itrack < nt; ++itrack) {
    double zrange = max(sel_zrange_ / sqrt(beta * gtracks.dz2[itrack]), zrange_min_);

    double zmin = gtracks.z[itrack] - zrange;
    unsigned int kmin = min(nv - 1, gtracks.kmin[itrack]);
    // find the smallest vertex_z that is larger than zmin
    if (gvertices.z_ptr[kmin] > zmin) {
      while ((kmin > 0) && (gvertices.z_ptr[kmin - 1] > zmin)) {
        kmin--;
      }
    } else {
      while ((kmin < (nv - 1)) && (gvertices.z_ptr[kmin] < zmin)) {
        kmin++;
      }
    }

    double zmax = gtracks.z[itrack] + zrange;
    unsigned int kmax = min(nv - 1, gtracks.kmax[itrack] - 1);
    // note: kmax points to the last vertex in the range, while gtracks.kmax points to the entry BEHIND the last vertex
    // find the largest vertex_z that is smaller than zmax
    if (gvertices.z_ptr[kmax] < zmax) {
      while ((kmax < (nv - 1)) && (gvertices.z_ptr[kmax + 1] < zmax)) {
        kmax++;
      }
    } else {
      while ((kmax > 0) && (gvertices.z_ptr[kmax] > zmax)) {
        kmax--;
      }
    }

    if (kmin <= kmax) {
      gtracks.kmin[itrack] = kmin;
      gtracks.kmax[itrack] = kmax + 1;
    } else {
      gtracks.kmin[itrack] = max(0U, min(kmin, kmax));
      gtracks.kmax[itrack] = min(nv, max(kmin, kmax) + 1);
    }
  }
}

void DAClusterizerInZ_vect::clear_vtx_range(track_t& gtracks, vertex_t& gvertices) const {
  const unsigned int nt = gtracks.getSize();
  const unsigned int nv = gvertices.getSize();
  for (auto itrack = 0U; itrack < nt; ++itrack) {
    gtracks.kmin[itrack] = 0;
    gtracks.kmax[itrack] = nv;
  }
}

double DAClusterizerInZ_vect::update(double beta, track_t& gtracks, vertex_t& gvertices, const double rho0) const {
  //update weights and vertex positions
  // mass constrained annealing without noise
  // returns the maximum of changes of vertex positions
  // identical to updateTC but without updating swE needed for Tc

  const unsigned int nt = gtracks.getSize();
  const unsigned int nv = gvertices.getSize();

  //initialize sums
  double sumpi = 0;

  // to return how much the prototype moved
  double delta = 0;

  // intial value of a sum
  double Z_init = 0;
  // independpent of loop
  if (rho0 > 0) {
    Z_init = rho0 * local_exp(-beta * dzCutOff_ * dzCutOff_);
  }

  // define kernels
  auto kernel_calc_exp_arg_range = [beta](const unsigned int itrack,
                                          track_t const& tracks,
                                          vertex_t const& vertices,
                                          const unsigned int kmin,
                                          const unsigned int kmax) {
    const double track_z = tracks.z_ptr[itrack];
    const double botrack_dz2 = -beta * tracks.dz2_ptr[itrack];

    // auto-vectorized
    for (unsigned int ivertex = kmin; ivertex < kmax; ++ivertex) {
      auto mult_res = track_z - vertices.z_ptr[ivertex];
      vertices.ei_cache_ptr[ivertex] = botrack_dz2 * (mult_res * mult_res);
    }
  };

  auto kernel_add_Z_range = [Z_init](
                                vertex_t const& vertices, const unsigned int kmin, const unsigned int kmax) -> double {
    double ZTemp = Z_init;
    for (unsigned int ivertex = kmin; ivertex < kmax; ++ivertex) {
      ZTemp += vertices.pk_ptr[ivertex] * vertices.ei_ptr[ivertex];
    }
    return ZTemp;
  };

  auto kernel_calc_normalization_range = [](const unsigned int track_num,
                                            track_t& tks_vec,
                                            vertex_t& y_vec,
                                            const unsigned int kmin,
                                            const unsigned int kmax) {
    auto tmp_trk_pi = tks_vec.pi_ptr[track_num];
    auto o_trk_Z_sum = 1. / tks_vec.Z_sum_ptr[track_num];
    auto o_trk_dz2 = tks_vec.dz2_ptr[track_num];
    auto tmp_trk_z = tks_vec.z_ptr[track_num];

    // auto-vectorized
    for (unsigned int k = kmin; k < kmax; ++k) {
      y_vec.se_ptr[k] += y_vec.ei_ptr[k] * (tmp_trk_pi * o_trk_Z_sum);
      auto w = y_vec.pk_ptr[k] * y_vec.ei_ptr[k] * (tmp_trk_pi * o_trk_Z_sum * o_trk_dz2);
      y_vec.sw_ptr[k] += w;
      y_vec.swz_ptr[k] += w * tmp_trk_z;
    }
  };

  for (auto ivertex = 0U; ivertex < nv; ++ivertex) {
    gvertices.se_ptr[ivertex] = 0.0;
    gvertices.sw_ptr[ivertex] = 0.0;
    gvertices.swz_ptr[ivertex] = 0.0;
  }

  // loop over tracks
  for (auto itrack = 0U; itrack < nt; ++itrack) {
    unsigned int kmin = gtracks.kmin[itrack];
    unsigned int kmax = gtracks.kmax[itrack];

#ifdef DEBUG
    assert((kmin < kmax) && (kmax <= nv));
    assert(itrack < gtracks.Z_sum.size());
#endif

    kernel_calc_exp_arg_range(itrack, gtracks, gvertices, kmin, kmax);
    local_exp_list_range(gvertices.ei_cache_ptr, gvertices.ei_ptr, kmin, kmax);
    gtracks.Z_sum_ptr[itrack] = kernel_add_Z_range(gvertices, kmin, kmax);

    if (edm::isNotFinite(gtracks.Z_sum_ptr[itrack]))
      gtracks.Z_sum_ptr[itrack] = 0.0;
    // used in the next major loop to follow
    sumpi += gtracks.pi_ptr[itrack];

    if (gtracks.Z_sum_ptr[itrack] > 1.e-100) {
      kernel_calc_normalization_range(itrack, gtracks, gvertices, kmin, kmax);
    }
  }

  // now update z and pk
  auto kernel_calc_z = [sumpi, nv](vertex_t& vertices) -> double {
    double delta = 0;
    // does not vectorize(?)
    for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
      if (vertices.sw_ptr[ivertex] > 0) {
        auto znew = vertices.swz_ptr[ivertex] / vertices.sw_ptr[ivertex];
        delta = max(std::abs(vertices.z_ptr[ivertex] - znew), delta);
        vertices.z_ptr[ivertex] = znew;
      }
    }

    auto osumpi = 1. / sumpi;
    for (unsigned int ivertex = 0; ivertex < nv; ++ivertex)
      vertices.pk_ptr[ivertex] = vertices.pk_ptr[ivertex] * vertices.se_ptr[ivertex] * osumpi;

    return delta;
  };

  delta = kernel_calc_z(gvertices);

  // return how much the prototypes moved
  return delta;
}

double DAClusterizerInZ_vect::updateTc(double beta, track_t& gtracks, vertex_t& gvertices, const double rho0) const {
  // update weights and vertex positions and Tc input
  // returns the squared sum of changes of vertex positions

  const unsigned int nt = gtracks.getSize();
  const unsigned int nv = gvertices.getSize();

  //initialize sums
  double sumpi = 0;

  // to return how much the prototype moved
  double delta = 0;

  // independpent of loop
  double Z_init = 0;
  if (rho0 > 0) {
    Z_init = rho0 * local_exp(-beta * dzCutOff_ * dzCutOff_);  // cut-off
  }

  // define kernels
  auto kernel_calc_exp_arg_range = [beta](const unsigned int itrack,
                                          track_t const& tracks,
                                          vertex_t const& vertices,
                                          const unsigned int kmin,
                                          const unsigned int kmax) {
    const double track_z = tracks.z_ptr[itrack];
    const double botrack_dz2 = -beta * tracks.dz2_ptr[itrack];

    // auto-vectorized
    for (unsigned int ivertex = kmin; ivertex < kmax; ++ivertex) {
      auto mult_res = track_z - vertices.z_ptr[ivertex];
      vertices.ei_cache_ptr[ivertex] = botrack_dz2 * (mult_res * mult_res);
    }
  };

  auto kernel_add_Z_range = [Z_init](
                                vertex_t const& vertices, const unsigned int kmin, const unsigned int kmax) -> double {
    double ZTemp = Z_init;
    for (unsigned int ivertex = kmin; ivertex < kmax; ++ivertex) {
      ZTemp += vertices.pk_ptr[ivertex] * vertices.ei_ptr[ivertex];
    }
    return ZTemp;
  };

  auto kernel_calc_normalization_range = [beta](const unsigned int track_num,
                                                track_t& tks_vec,
                                                vertex_t& y_vec,
                                                const unsigned int kmin,
                                                const unsigned int kmax) {
    auto tmp_trk_pi = tks_vec.pi_ptr[track_num];
    auto o_trk_Z_sum = 1. / tks_vec.Z_sum_ptr[track_num];
    auto o_trk_dz2 = tks_vec.dz2_ptr[track_num];
    auto tmp_trk_z = tks_vec.z_ptr[track_num];
    auto obeta = -1. / beta;

    // auto-vectorized
    for (unsigned int k = kmin; k < kmax; ++k) {
      y_vec.se_ptr[k] += y_vec.ei_ptr[k] * (tmp_trk_pi * o_trk_Z_sum);
      auto w = y_vec.pk_ptr[k] * y_vec.ei_ptr[k] * (tmp_trk_pi * o_trk_Z_sum * o_trk_dz2);
      y_vec.sw_ptr[k] += w;
      y_vec.swz_ptr[k] += w * tmp_trk_z;
      y_vec.swE_ptr[k] += w * y_vec.ei_cache_ptr[k] * obeta;
    }
  };

  for (auto ivertex = 0U; ivertex < nv; ++ivertex) {
    gvertices.se_ptr[ivertex] = 0.0;
    gvertices.sw_ptr[ivertex] = 0.0;
    gvertices.swz_ptr[ivertex] = 0.0;
    gvertices.swE_ptr[ivertex] = 0.0;
  }

  // loop over tracks
  for (auto itrack = 0U; itrack < nt; ++itrack) {
    unsigned int kmin = gtracks.kmin[itrack];
    unsigned int kmax = gtracks.kmax[itrack];

    kernel_calc_exp_arg_range(itrack, gtracks, gvertices, kmin, kmax);
    local_exp_list_range(gvertices.ei_cache_ptr, gvertices.ei_ptr, kmin, kmax);
    gtracks.Z_sum_ptr[itrack] = kernel_add_Z_range(gvertices, kmin, kmax);

    if (edm::isNotFinite(gtracks.Z_sum_ptr[itrack]))
      gtracks.Z_sum_ptr[itrack] = 0.0;
    // used in the next major loop to follow
    sumpi += gtracks.pi_ptr[itrack];

    if (gtracks.Z_sum_ptr[itrack] > 1.e-100) {
      kernel_calc_normalization_range(itrack, gtracks, gvertices, kmin, kmax);
    }
  }

  // now update z and pk
  auto kernel_calc_z = [sumpi, nv](vertex_t& vertices) -> double {
    double delta = 0;
    // does not vectorizes
    for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
      if (vertices.sw_ptr[ivertex] > 0) {
        auto znew = vertices.swz_ptr[ivertex] / vertices.sw_ptr[ivertex];
        // prevents from vectorizing if
        delta = max(std::abs(vertices.z_ptr[ivertex] - znew), delta);
        vertices.z_ptr[ivertex] = znew;
      }
    }

    auto osumpi = 1. / sumpi;
    for (unsigned int ivertex = 0; ivertex < nv; ++ivertex)
      vertices.pk_ptr[ivertex] = vertices.pk_ptr[ivertex] * vertices.se_ptr[ivertex] * osumpi;

    return delta;
  };

  delta = kernel_calc_z(gvertices);

  // return how much the prototypes moved
  return delta;
}

double DAClusterizerInZ_vect::evalF(const double beta, track_t const& tks, vertex_t const& v) const {
  // temporary : evaluate the original F
  auto nt = tks.getSize();
  auto nv = v.getSize();
  double F = 0;
  for (auto i = 0U; i < nt; i++) {
    double Z = 0;
    for (auto k = 0u; k < nv; k++) {
      double Eik = (tks.z[k] - v.z[i]) * (tks.z[k] - v.z[i]) * tks.dz2[i];
      if ((beta * Eik) < 30) {
        Z += v.pk[k] * local_exp(-beta * Eik);
      }
    }
    if (Z > 0) {
      F += tks.pi[i] * log(Z);
    }
  }
  std::cout << "F(full) = " << -F / beta << std::endl;
  return -F / beta;
}

unsigned int DAClusterizerInZ_vect::thermalize(
    double beta, track_t& tks, vertex_t& v, const double delta_max0, const double rho0) const {
  unsigned int niter = 0;
  double delta = 0;
  double delta_max = delta_lowT_;

  if (convergence_mode_ == 0) {
    delta_max = delta_max0;
  } else if (convergence_mode_ == 1) {
    delta_max = delta_lowT_ / sqrt(std::max(beta, 1.0));
  }

  set_vtx_range(beta, tks, v);
  double delta_sum_range = 0;  // accumulate max(|delta-z|) as a lower bound
  std::vector<double> z0 = v.z;

  while (niter++ < maxIterations_) {
    delta = update(beta, tks, v, rho0);
    delta_sum_range += delta;

    if (delta_sum_range > zrange_min_) {
      for (unsigned int k = 0; k < v.getSize(); k++) {
        if (std::abs(v.z[k] - z0[k]) > zrange_min_) {
          set_vtx_range(beta, tks, v);
          delta_sum_range = 0;
          z0 = v.z;
          break;
        }
      }
    }

    if (delta < delta_max) {
      break;
    }
  }

#ifdef DEBUG
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZ_vect.thermalize niter = " << niter << " at T = " << 1 / beta
              << "  nv = " << v.getSize() << std::endl;
    if (DEBUGLEVEL > 2)
      dump(beta, v, tks, 0);
  }
#endif

  return niter;
}

bool DAClusterizerInZ_vect::merge(vertex_t& y, track_t& tks, double& beta) const {
  // merge clusters that collapsed or never separated,
  // only merge if the estimated critical temperature of the merged vertex is below the current temperature
  // return true if vertices were merged, false otherwise
  const unsigned int nv = y.getSize();

  if (nv < 2)
    return false;

  // merge the smallest distance clusters first
  std::vector<std::pair<double, unsigned int> > critical;
  for (unsigned int k = 0; (k + 1) < nv; k++) {
    if (std::fabs(y.z_ptr[k + 1] - y.z_ptr[k]) < zmerge_) {
      critical.push_back(make_pair(std::fabs(y.z_ptr[k + 1] - y.z_ptr[k]), k));
    }
  }
  if (critical.empty())
    return false;

  std::stable_sort(critical.begin(), critical.end(), std::less<std::pair<double, unsigned int> >());

  for (unsigned int ik = 0; ik < critical.size(); ik++) {
    unsigned int k = critical[ik].second;
    double rho = y.pk_ptr[k] + y.pk_ptr[k + 1];
    double swE = y.swE_ptr[k] + y.swE_ptr[k + 1] -
                 y.pk_ptr[k] * y.pk_ptr[k + 1] / rho * std::pow(y.z_ptr[k + 1] - y.z_ptr[k], 2);
    double Tc = 2 * swE / (y.sw_ptr[k] + y.sw_ptr[k + 1]);

    if (Tc * beta < 1) {
#ifdef DEBUG
      assert((k + 1) < nv);
      if (DEBUGLEVEL > 1) {
        std::cout << "merging " << fixed << setprecision(4) << y.z_ptr[k + 1] << " and " << y.z_ptr[k]
                  << "  Tc = " << Tc << "  sw = " << y.sw_ptr[k] + y.sw_ptr[k + 1] << std::endl;
      }
#endif

      if (rho > 0) {
        y.z_ptr[k] = (y.pk_ptr[k] * y.z_ptr[k] + y.pk_ptr[k + 1] * y.z_ptr[k + 1]) / rho;
      } else {
        y.z_ptr[k] = 0.5 * (y.z_ptr[k] + y.z_ptr[k + 1]);
      }
      y.pk_ptr[k] = rho;
      y.sw_ptr[k] += y.sw_ptr[k + 1];
      y.swE_ptr[k] = swE;
      y.removeItem(k + 1, tks);
      set_vtx_range(beta, tks, y);
      y.extractRaw();
      return true;
    }
  }

  return false;
}

bool DAClusterizerInZ_vect::purge(vertex_t& y, track_t& tks, double& rho0, const double beta) const {
  constexpr double eps = 1.e-100;
  // eliminate clusters with only one significant/unique track
  const unsigned int nv = y.getSize();
  const unsigned int nt = tks.getSize();

  if (nv < 2)
    return false;

  double sumpmin = nt;
  unsigned int k0 = nv;

  std::vector<double> inverse_zsums(nt), arg_cache(nt), eik_cache(nt), pcut_cache(nv);
  double* __restrict__ pinverse_zsums;
  double* __restrict__ parg_cache;
  double* __restrict__ peik_cache;
  double* __restrict__ ppcut_cache;
  pinverse_zsums = inverse_zsums.data();
  parg_cache = arg_cache.data();
  peik_cache = eik_cache.data();
  ppcut_cache = pcut_cache.data();
  for (unsigned i = 0; i < nt; ++i) {
    inverse_zsums[i] = tks.Z_sum_ptr[i] > eps ? 1. / tks.Z_sum_ptr[i] : 0.0;
  }
  const auto rhoconst = rho0 * local_exp(-beta * dzCutOff_ * dzCutOff_);
  for (unsigned int k = 0; k < nv; k++) {
    const double pmax = y.pk_ptr[k] / (y.pk_ptr[k] + rhoconst);
    ppcut_cache[k] = uniquetrkweight_ * pmax;
  }

  for (unsigned int k = 0; k < nv; k++) {
    for (unsigned int i = 0; i < nt; ++i) {
      const auto track_z = tks.z_ptr[i];
      const auto botrack_dz2 = -beta * tks.dz2_ptr[i];
      const auto mult_resz = track_z - y.z_ptr[k];
      parg_cache[i] = botrack_dz2 * (mult_resz * mult_resz);
    }
    local_exp_list(parg_cache, peik_cache, nt);

    int nUnique = 0;
    double sump = 0;
    for (unsigned int i = 0; i < nt; ++i) {
      const auto p = y.pk_ptr[k] * peik_cache[i] * pinverse_zsums[i];
      sump += p;
      nUnique += ((p > ppcut_cache[k]) & (tks.pi_ptr[i] > 0)) ? 1 : 0;
    }

    if ((nUnique < 2) && (sump < sumpmin)) {
      sumpmin = sump;
      k0 = k;
    }
  }

  if (k0 != nv) {
#ifdef DEBUG
    assert(k0 < y.getSize());
    if (DEBUGLEVEL > 1) {
      std::cout << "eliminating prototype at " << std::setw(10) << std::setprecision(4) << y.z_ptr[k0]
                << " with sump=" << sumpmin << "  rho*nt =" << y.pk_ptr[k0] * nt << endl;
    }
#endif

    y.removeItem(k0, tks);
    set_vtx_range(beta, tks, y);
    return true;
  } else {
    return false;
  }
}

double DAClusterizerInZ_vect::beta0(double betamax, track_t const& tks, vertex_t const& y) const {
  double T0 = 0;  // max Tc for beta=0
  // estimate critical temperature from beta=0 (T=inf)
  const unsigned int nt = tks.getSize();
  const unsigned int nv = y.getSize();

  for (unsigned int k = 0; k < nv; k++) {
    // vertex fit at T=inf
    double sumwz = 0;
    double sumw = 0;
    for (unsigned int i = 0; i < nt; i++) {
      double w = tks.pi_ptr[i] * tks.dz2_ptr[i];
      sumwz += w * tks.z_ptr[i];
      sumw += w;
    }

    y.z_ptr[k] = sumwz / sumw;

    // estimate Tcrit
    double a = 0, b = 0;
    for (unsigned int i = 0; i < nt; i++) {
      double dx = tks.z_ptr[i] - y.z_ptr[k];
      double w = tks.pi_ptr[i] * tks.dz2_ptr[i];
      a += w * std::pow(dx, 2) * tks.dz2_ptr[i];
      b += w;
    }
    double Tc = 2. * a / b;  // the critical temperature of this vertex

    if (Tc > T0)
      T0 = Tc;

  }  // vertex loop (normally there should be only one vertex at beta=0)

#ifdef DEBUG
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZ_vect.beta0:   Tc = " << T0 << std::endl;
    int coolingsteps = 1 - int(std::log(T0 * betamax) / std::log(coolingFactor_));
    std::cout << "DAClusterizerInZ_vect.beta0:   nstep = " << coolingsteps << std::endl;
  }
#endif

  if (T0 > 1. / betamax) {
    int coolingsteps = 1 - int(std::log(T0 * betamax) / std::log(coolingFactor_));

    return betamax * std::pow(coolingFactor_, coolingsteps);
  } else {
    // ensure at least one annealing step
    return betamax * coolingFactor_;
  }
}

bool DAClusterizerInZ_vect::split(const double beta, track_t& tks, vertex_t& y, double threshold) const {
  // split only critical vertices (Tc >~ T=1/beta   <==>   beta*Tc>~1)
  // an update must have been made just before doing this (same beta, no merging)
  // returns true if at least one cluster was split

  constexpr double epsilon = 1e-3;  // minimum split size
  unsigned int nv = y.getSize();

  // avoid left-right biases by splitting highest Tc first

  std::vector<std::pair<double, unsigned int> > critical;
  for (unsigned int k = 0; k < nv; k++) {
    double Tc = 2 * y.swE_ptr[k] / y.sw_ptr[k];
    if (beta * Tc > threshold) {
      critical.push_back(make_pair(Tc, k));
    }
  }
  if (critical.empty())
    return false;

  std::stable_sort(critical.begin(), critical.end(), std::greater<std::pair<double, unsigned int> >());

  bool split = false;
  const unsigned int nt = tks.getSize();

  for (unsigned int ic = 0; ic < critical.size(); ic++) {
    unsigned int k = critical[ic].second;

    // estimate subcluster positions and weight
    double p1 = 0, z1 = 0, w1 = 0;
    double p2 = 0, z2 = 0, w2 = 0;
    for (unsigned int i = 0; i < nt; i++) {
      if (tks.Z_sum_ptr[i] > 1.e-100) {
        // winner-takes-all, usually overestimates splitting
        double tl = tks.z_ptr[i] < y.z_ptr[k] ? 1. : 0.;
        double tr = 1. - tl;

        // soften it, especially at low T
        double arg = (tks.z_ptr[i] - y.z_ptr[k]) * sqrt(beta * tks.dz2_ptr[i]);
        if (std::fabs(arg) < 20) {
          double t = local_exp(-arg);
          tl = t / (t + 1.);
          tr = 1 / (t + 1.);
        }

        double p = y.pk_ptr[k] * tks.pi_ptr[i] * local_exp(-beta * Eik(tks.z_ptr[i], y.z_ptr[k], tks.dz2_ptr[i])) /
                   tks.Z_sum_ptr[i];
        double w = p * tks.dz2_ptr[i];
        p1 += p * tl;
        z1 += w * tl * tks.z_ptr[i];
        w1 += w * tl;
        p2 += p * tr;
        z2 += w * tr * tks.z_ptr[i];
        w2 += w * tr;
      }
    }

    if (w1 > 0) {
      z1 = z1 / w1;
    } else {
      z1 = y.z_ptr[k] - epsilon;
    }
    if (w2 > 0) {
      z2 = z2 / w2;
    } else {
      z2 = y.z_ptr[k] + epsilon;
    }

    // reduce split size if there is not enough room
    if ((k > 0) && (z1 < (0.6 * y.z_ptr[k] + 0.4 * y.z_ptr[k - 1]))) {
      z1 = 0.6 * y.z_ptr[k] + 0.4 * y.z_ptr[k - 1];
    }
    if ((k + 1 < nv) && (z2 > (0.6 * y.z_ptr[k] + 0.4 * y.z_ptr[k + 1]))) {
      z2 = 0.6 * y.z_ptr[k] + 0.4 * y.z_ptr[k + 1];
    }

#ifdef DEBUG
    assert(k < nv);
    if (DEBUGLEVEL > 1) {
      if (std::fabs(y.z_ptr[k] - zdumpcenter_) < zdumpwidth_) {
        std::cout << " T= " << std::setw(8) << 1. / beta << " Tc= " << critical[ic].first << "    splitting "
                  << std::fixed << std::setprecision(4) << y.z_ptr[k] << " --> " << z1 << "," << z2 << "     [" << p1
                  << "," << p2 << "]";
        if (std::fabs(z2 - z1) > epsilon) {
          std::cout << std::endl;
        } else {
          std::cout << "  rejected " << std::endl;
        }
      }
    }
#endif

    // split if the new subclusters are significantly separated
    if ((z2 - z1) > epsilon) {
      split = true;
      double pk1 = p1 * y.pk_ptr[k] / (p1 + p2);
      double pk2 = p2 * y.pk_ptr[k] / (p1 + p2);
      y.z_ptr[k] = z2;
      y.pk_ptr[k] = pk2;
      y.insertItem(k, z1, pk1, tks);
      if (k == 0)
        y.extractRaw();

      nv++;

      // adjust remaining pointers
      for (unsigned int jc = ic; jc < critical.size(); jc++) {
        if (critical[jc].second >= k) {
          critical[jc].second++;
        }
      }
    }
  }

  return split;
}

vector<TransientVertex> DAClusterizerInZ_vect::vertices(const vector<reco::TransientTrack>& tracks,
                                                        const int verbosity) const {
  track_t&& tks = fill(tracks);
  tks.extractRaw();

  unsigned int nt = tks.getSize();
  double rho0 = 0.0;  // start with no outlier rejection

  vector<TransientVertex> clusters;
  if (tks.getSize() == 0)
    return clusters;

  vertex_t y;  // the vertex prototypes

  // initialize:single vertex at infinite temperature
  y.addItem(0, 1.0);
  clear_vtx_range(tks, y);

  // estimate first critical temperature
  double beta = beta0(betamax_, tks, y);
#ifdef DEBUG
  if (DEBUGLEVEL > 0)
    std::cout << "Beta0 is " << beta << std::endl;
#endif

  thermalize(beta, tks, y, delta_highT_);

  // annealing loop, stop when T<Tmin  (i.e. beta>1/Tmin)

  double betafreeze = betamax_ * sqrt(coolingFactor_);

  while (beta < betafreeze) {
    updateTc(beta, tks, y, rho0);
    while (merge(y, tks, beta)) {
      updateTc(beta, tks, y, rho0);
    }
    split(beta, tks, y);

    beta = beta / coolingFactor_;
    set_vtx_range(beta, tks, y);
    thermalize(beta, tks, y, delta_highT_);
  }

#ifdef DEBUG
  verify(y, tks);

  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZ_vect::vertices :"
              << "last round of splitting" << std::endl;
  }
#endif

  set_vtx_range(beta, tks, y);
  updateTc(beta, tks, y, rho0);  // make sure Tc is up-to-date

  while (merge(y, tks, beta)) {
    set_vtx_range(beta, tks, y);
    updateTc(beta, tks, y, rho0);
  }

  unsigned int ntry = 0;
  double threshold = 1.0;
  while (split(beta, tks, y, threshold) && (ntry++ < 10)) {
    set_vtx_range(beta, tks, y);
    thermalize(beta, tks, y, delta_highT_, 0.);
    updateTc(beta, tks, y, rho0);
    while (merge(y, tks, beta)) {
      updateTc(beta, tks, y, rho0);
    }

    // relax splitting a bit to reduce multiple split-merge cycles of the same cluster
    threshold *= 1.1;
  }

#ifdef DEBUG
  verify(y, tks);
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZ_vect::vertices :"
              << "turning on outlier rejection at T=" << 1 / beta << std::endl;
  }
#endif

  // switch on outlier rejection at T=Tmin, doesn't do much at high PU
  if (dzCutOff_ > 0) {
    rho0 = 1. / nt;  //1. / y.getSize();??
    for (unsigned int a = 0; a < 5; a++) {
      update(beta, tks, y, a * rho0 / 5.);  // adiabatic turn-on
    }
  }

  thermalize(beta, tks, y, delta_lowT_, rho0);

#ifdef DEBUG
  verify(y, tks);
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZ_vect::vertices :"
              << "merging with outlier rejection at T=" << 1 / beta << std::endl;
  }
  if (DEBUGLEVEL > 2)
    dump(beta, y, tks, 2);
#endif

  // merge again  (some cluster split by outliers collapse here)
  while (merge(y, tks, beta)) {
    set_vtx_range(beta, tks, y);
    update(beta, tks, y, rho0);
  }

#ifdef DEBUG
  verify(y, tks);
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZ_vect::vertices :"
              << "after merging with outlier rejection at T=" << 1 / beta << std::endl;
  }
  if (DEBUGLEVEL > 2)
    dump(beta, y, tks, 2);
#endif

  // go down to the purging temperature (if it is lower than tmin)
  while (beta < betapurge_) {
    beta = min(beta / coolingFactor_, betapurge_);
    set_vtx_range(beta, tks, y);
    thermalize(beta, tks, y, delta_lowT_, rho0);
  }

#ifdef DEBUG
  verify(y, tks);
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZ_vect::vertices :"
              << "purging at T=" << 1 / beta << std::endl;
  }
#endif

  // eliminate insigificant vertices, this is more restrictive at higher T
  while (purge(y, tks, rho0, beta)) {
    thermalize(beta, tks, y, delta_lowT_, rho0);
  }

#ifdef DEBUG
  verify(y, tks);
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZ_vect::vertices :"
              << "last cooling T=" << 1 / beta << std::endl;
  }
#endif

  // optionally cool some more without doing anything, to make the assignment harder
  while (beta < betastop_) {
    beta = min(beta / coolingFactor_, betastop_);
    thermalize(beta, tks, y, delta_lowT_, rho0);
  }

#ifdef DEBUG
  verify(y, tks);
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZ_vect::vertices :"
              << "stop cooling at T=" << 1 / beta << std::endl;
  }
  if (DEBUGLEVEL > 2)
    dump(beta, y, tks, 2);
#endif

  // select significant tracks and use a TransientVertex as a container
  GlobalError dummyError(0.01, 0, 0.01, 0., 0., 0.01);

  // ensure correct normalization of probabilities, should makes double assignment reasonably impossible
  const unsigned int nv = y.getSize();
  for (unsigned int k = 0; k < nv; k++)
    if (edm::isNotFinite(y.pk_ptr[k]) || edm::isNotFinite(y.z_ptr[k])) {
      y.pk_ptr[k] = 0;
      y.z_ptr[k] = 0;
    }

  const auto z_sum_init = rho0 * local_exp(-beta * dzCutOff_ * dzCutOff_);
  for (unsigned int i = 0; i < nt; i++)  // initialize
    tks.Z_sum_ptr[i] = z_sum_init;

  // improve vectorization (does not require reduction ....)
  for (unsigned int k = 0; k < nv; k++) {
    for (unsigned int i = 0; i < nt; i++)
      tks.Z_sum_ptr[i] += y.pk_ptr[k] * local_exp(-beta * Eik(tks.z_ptr[i], y.z_ptr[k], tks.dz2_ptr[i]));
  }

  for (unsigned int k = 0; k < nv; k++) {
    GlobalPoint pos(0, 0, y.z_ptr[k]);

    vector<reco::TransientTrack> vertexTracks;
    for (unsigned int i = 0; i < nt; i++) {
      if (tks.Z_sum_ptr[i] > 1e-100) {
        double p = y.pk_ptr[k] * local_exp(-beta * Eik(tks.z_ptr[i], y.z_ptr[k], tks.dz2_ptr[i])) / tks.Z_sum_ptr[i];
        if ((tks.pi_ptr[i] > 0) && (p > mintrkweight_)) {
          vertexTracks.push_back(*(tks.tt[i]));
          tks.Z_sum_ptr[i] = 0;  // setting Z=0 excludes double assignment
        }
      }
    }
    TransientVertex v(pos, dummyError, vertexTracks, 0);
    clusters.push_back(v);
  }

  return clusters;
}

vector<vector<reco::TransientTrack> > DAClusterizerInZ_vect::clusterize(
    const vector<reco::TransientTrack>& tracks) const {
  vector<vector<reco::TransientTrack> > clusters;
  vector<TransientVertex>&& pv = vertices(tracks);

#ifdef DEBUG
  if (DEBUGLEVEL > 0) {
    std::cout << "###################################################" << endl;
    std::cout << "# vectorized DAClusterizerInZ_vect::clusterize   nt=" << tracks.size() << endl;
    std::cout << "# DAClusterizerInZ_vect::clusterize   pv.size=" << pv.size() << endl;
    std::cout << "###################################################" << endl;
  }
#endif

  if (pv.empty()) {
    return clusters;
  }

  // fill into clusters and merge
  vector<reco::TransientTrack> aCluster = pv.begin()->originalTracks();

  for (auto k = pv.begin() + 1; k != pv.end(); k++) {
    if (std::abs(k->position().z() - (k - 1)->position().z()) > (2 * vertexSize_)) {
      // close a cluster
      if (aCluster.size() > 1) {
        clusters.push_back(aCluster);
      }
#ifdef DEBUG
      else {
        std::cout << " one track cluster at " << k->position().z() << "  suppressed" << std::endl;
      }
#endif
      aCluster.clear();
    }
    for (unsigned int i = 0; i < k->originalTracks().size(); i++) {
      aCluster.push_back(k->originalTracks()[i]);
    }
  }
  clusters.emplace_back(std::move(aCluster));

  return clusters;
}

void DAClusterizerInZ_vect::dump(const double beta, const vertex_t& y, const track_t& tks, int verbosity) const {
#ifdef DEBUG
  const unsigned int nv = y.getSize();
  const unsigned int nt = tks.getSize();

  std::vector<unsigned int> iz;
  for (unsigned int j = 0; j < nt; j++) {
    iz.push_back(j);
  }
  std::sort(iz.begin(), iz.end(), [tks](unsigned int a, unsigned int b) { return tks.z_ptr[a] < tks.z_ptr[b]; });
  std::cout << std::endl;
  std::cout << "-----DAClusterizerInZ::dump ----" << nv << "  clusters " << std::endl;
  std::cout << "                                                                   ";
  for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
    if (std::fabs(y.z_ptr[ivertex] - zdumpcenter_) < zdumpwidth_) {
      std::cout << "   " << setw(3) << ivertex << "  ";
    }
  }
  std::cout << endl;
  std::cout << "                                                                z= ";
  std::cout << setprecision(4);
  for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
    if (std::fabs(y.z_ptr[ivertex] - zdumpcenter_) < zdumpwidth_) {
      std::cout << setw(8) << fixed << y.z_ptr[ivertex];
    }
  }
  std::cout << endl
            << "T=" << setw(15) << 1. / beta << " Tmin =" << setw(10) << 1. / betamax_
            << "                             Tc= ";
  for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
    if (std::fabs(y.z_ptr[ivertex] - zdumpcenter_) < zdumpwidth_) {
      double Tc = 2 * y.swE_ptr[ivertex] / y.sw_ptr[ivertex];
      std::cout << setw(8) << fixed << setprecision(1) << Tc;
    }
  }
  std::cout << endl;

  std::cout << "                                                               pk= ";
  double sumpk = 0;
  for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
    sumpk += y.pk_ptr[ivertex];
    if (std::fabs(y.z_ptr[ivertex] - zdumpcenter_) > zdumpwidth_)
      continue;
    std::cout << setw(8) << setprecision(4) << fixed << y.pk_ptr[ivertex];
  }
  std::cout << endl;

  std::cout << "                                                               nt= ";
  for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
    if (std::fabs(y.z_ptr[ivertex] - zdumpcenter_) > zdumpwidth_)
      continue;
    std::cout << setw(8) << setprecision(1) << fixed << y.pk_ptr[ivertex] * nt;
  }
  std::cout << endl;

  if (verbosity > 0) {
    double E = 0, F = 0;
    std::cout << endl;
    std::cout << "----        z +/- dz                ip +/-dip       pt    phi  eta    weights  ----" << endl;
    std::cout << setprecision(4);
    for (unsigned int i0 = 0; i0 < nt; i0++) {
      unsigned int i = iz[i0];
      if (tks.Z_sum_ptr[i] > 0) {
        F -= std::log(tks.Z_sum_ptr[i]) / beta;
      }
      double tz = tks.z_ptr[i];

      if (std::fabs(tz - zdumpcenter_) > zdumpwidth_)
        continue;
      std::cout << setw(4) << i << ")" << setw(8) << fixed << setprecision(4) << tz << " +/-" << setw(6)
                << sqrt(1. / tks.dz2_ptr[i]);
      if ((tks.tt[i] == nullptr)) {
        std::cout << "          effective track                             ";
      } else {
        if (tks.tt[i]->track().quality(reco::TrackBase::highPurity)) {
          std::cout << " *";
        } else {
          std::cout << "  ";
        }
        if (tks.tt[i]->track().hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1)) {
          std::cout << "+";
        } else {
          std::cout << "-";
        }
        std::cout << setw(1)
                  << tks.tt[i]
                         ->track()
                         .hitPattern()
                         .pixelBarrelLayersWithMeasurement();  // see DataFormats/TrackReco/interface/HitPattern.h
        std::cout << setw(1) << tks.tt[i]->track().hitPattern().pixelEndcapLayersWithMeasurement();
        std::cout << setw(1) << hex
                  << tks.tt[i]->track().hitPattern().trackerLayersWithMeasurement() -
                         tks.tt[i]->track().hitPattern().pixelLayersWithMeasurement()
                  << dec;
        std::cout << "=" << setw(1) << hex
                  << tks.tt[i]->track().hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS) << dec;

        Measurement1D IP = tks.tt[i]->stateAtBeamLine().transverseImpactParameter();
        std::cout << setw(8) << IP.value() << "+/-" << setw(6) << IP.error();
        std::cout << " " << setw(6) << setprecision(2) << tks.tt[i]->track().pt() * tks.tt[i]->track().charge();
        std::cout << " " << setw(5) << setprecision(2) << tks.tt[i]->track().phi() << " " << setw(5) << setprecision(2)
                  << tks.tt[i]->track().eta();
      }  // not a dummy track

      double sump = 0.;
      for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
        if (std::fabs(y.z_ptr[ivertex] - zdumpcenter_) > zdumpwidth_)
          continue;

        if ((tks.pi_ptr[i] > 0) && (tks.Z_sum_ptr[i] > 0)) {
          //double p=pik(beta,tks[i],*k);
          double p = y.pk_ptr[ivertex] * local_exp(-beta * Eik(tks.z_ptr[i], y.z_ptr[ivertex], tks.dz2_ptr[i])) /
                     tks.Z_sum_ptr[i];
          if (p > 0.0001) {
            std::cout << setw(8) << setprecision(3) << p;
          } else {
            std::cout << "    .   ";
          }
          E += p * Eik(tks.z_ptr[i], y.z_ptr[ivertex], tks.dz2_ptr[i]);
          sump += p;
        } else {
          std::cout << "        ";
        }
      }
      std::cout << "  ( " << std::setw(3) << tks.kmin[i] << "," << std::setw(3) << tks.kmax[i] - 1 << " ) ";
      std::cout << endl;
    }
    std::cout << "                                                                   ";
    for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
      if (std::fabs(y.z_ptr[ivertex] - zdumpcenter_) < zdumpwidth_) {
        std::cout << "   " << setw(3) << ivertex << "  ";
      }
    }
    std::cout << endl;
    std::cout << "                                                                z= ";
    std::cout << setprecision(4);
    for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
      if (std::fabs(y.z_ptr[ivertex] - zdumpcenter_) < zdumpwidth_) {
        std::cout << setw(8) << fixed << y.z_ptr[ivertex];
      }
    }
    std::cout << endl;
    std::cout << endl
              << "T=" << 1 / beta << " E=" << E << " n=" << y.getSize() << "  F= " << F << endl
              << "----------" << endl;
  }
#endif
}
