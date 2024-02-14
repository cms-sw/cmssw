#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZT_vect.h"
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

using namespace std;

//#define DEBUG
#ifdef DEBUG
#define DEBUGLEVEL 0
#endif

DAClusterizerInZT_vect::DAClusterizerInZT_vect(const edm::ParameterSet& conf) {
  // hardcoded parameters
  maxIterations_ = 1000;
  mintrkweight_ = 0.5;

  // configurable debug output
#ifdef DEBUG
  zdumpcenter_ = conf.getUntrackedParameter<double>("zdumpcenter", 0.);
  zdumpwidth_ = conf.getUntrackedParameter<double>("zdumpwidth", 20.);
#endif

  // configurable parameters
  double minT = conf.getParameter<double>("Tmin");
  double purgeT = conf.getParameter<double>("Tpurge");
  double stopT = conf.getParameter<double>("Tstop");
  vertexSize_ = conf.getParameter<double>("vertexSize");
  vertexSizeTime_ = conf.getParameter<double>("vertexSizeTime");
  coolingFactor_ = conf.getParameter<double>("coolingFactor");
  d0CutOff_ = conf.getParameter<double>("d0CutOff");
  dzCutOff_ = conf.getParameter<double>("dzCutOff");
  dtCutOff_ = conf.getParameter<double>("dtCutOff");
  t0Max_ = conf.getParameter<double>("t0Max");
  uniquetrkweight_ = conf.getParameter<double>("uniquetrkweight");
  uniquetrkminp_ = conf.getParameter<double>("uniquetrkminp");
  zmerge_ = conf.getParameter<double>("zmerge");
  tmerge_ = conf.getParameter<double>("tmerge");
  sel_zrange_ = conf.getParameter<double>("zrange");
  convergence_mode_ = conf.getParameter<int>("convergence_mode");
  delta_lowT_ = conf.getParameter<double>("delta_lowT");
  delta_highT_ = conf.getParameter<double>("delta_highT");

#ifdef DEBUG
  std::cout << "DAClusterizerInZT_vect: mintrkweight = " << mintrkweight_ << std::endl;
  std::cout << "DAClusterizerInZT_vect: uniquetrkweight = " << uniquetrkweight_ << std::endl;
  std::cout << "DAClusterizerInZT_vect: uniquetrkminp = " << uniquetrkminp_ << std::endl;
  std::cout << "DAClusterizerInZT_vect: zmerge = " << zmerge_ << std::endl;
  std::cout << "DAClusterizerInZT_vect: tmerge = " << tmerge_ << std::endl;
  std::cout << "DAClusterizerInZT_vect: Tmin = " << minT << std::endl;
  std::cout << "DAClusterizerInZT_vect: Tpurge = " << purgeT << std::endl;
  std::cout << "DAClusterizerInZT_vect: Tstop = " << stopT << std::endl;
  std::cout << "DAClusterizerInZT_vect: vertexSize = " << vertexSize_ << std::endl;
  std::cout << "DAClusterizerInZT_vect: vertexSizeTime = " << vertexSizeTime_ << std::endl;
  std::cout << "DAClusterizerInZT_vect: coolingFactor = " << coolingFactor_ << std::endl;
  std::cout << "DAClusterizerInZT_vect: d0CutOff = " << d0CutOff_ << std::endl;
  std::cout << "DAClusterizerInZT_vect: dzCutOff = " << dzCutOff_ << std::endl;
  std::cout << "DAClusterizerInZT_vect: dtCutoff = " << dtCutOff_ << std::endl;
  std::cout << "DAClusterizerInZT_vect: zrange = " << sel_zrange_ << std::endl;
  std::cout << "DAClusterizerinZT_vect: convergence mode = " << convergence_mode_ << std::endl;
  std::cout << "DAClusterizerinZT_vect: delta_highT = " << delta_highT_ << std::endl;
  std::cout << "DAClusterizerinZT_vect: delta_lowT = " << delta_lowT_ << std::endl;
  std::cout << "DAClusterizerinZT_vect: DEBUGLEVEL " << DEBUGLEVEL << std::endl;
#endif

  if (convergence_mode_ > 1) {
    edm::LogWarning("DAClusterizerinZT_vect")
        << "DAClusterizerInZT_vect: invalid convergence_mode " << convergence_mode_ << "  reset to default " << 0;
    convergence_mode_ = 0;
  }

  if (minT == 0) {
    betamax_ = 1.0;
    edm::LogWarning("DAClusterizerinZT_vect")
        << "DAClusterizerInZT_vect: invalid Tmin " << minT << "  reset to default " << 1. / betamax_;
  } else {
    betamax_ = 1. / minT;
  }

  if ((purgeT > minT) || (purgeT == 0)) {
    edm::LogWarning("DAClusterizerinZT_vect")
        << "DAClusterizerInZT_vect: invalid Tpurge " << purgeT << "  set to " << minT;
    purgeT = minT;
  }
  betapurge_ = 1. / purgeT;

  if ((stopT > purgeT) || (stopT == 0)) {
    edm::LogWarning("DAClusterizerinZT_vect")
        << "DAClusterizerInZT_vect: invalid Tstop " << stopT << "  set to  " << max(1., purgeT);
    stopT = max(1., purgeT);
  }
  betastop_ = 1. / stopT;
}

namespace {
  inline double local_exp(double const& inp) { return vdt::fast_exp(inp); }

  inline void local_exp_list_range(double const* __restrict__ arg_inp,
                                   double* __restrict__ arg_out,
                                   const int kmin,
                                   const int kmax) {
    for (auto i = kmin; i != kmax; ++i)
      arg_out[i] = vdt::fast_exp(arg_inp[i]);
  }

}  // namespace

void DAClusterizerInZT_vect::verify(const vertex_t& v, const track_t& tks, unsigned int nv, unsigned int nt) const {
  if (nv == 999999) {
    nv = v.getSize();
  } else {
    assert(nv == v.getSize());
  }

  if (nt == 999999) {
    nt = tks.getSize();
  } else {
    assert(nt == tks.getSize());
  }

  // clusters
  assert(v.zvtx_vec.size() == nv);
  assert(v.tvtx_vec.size() == nv);
#ifdef USEVTXDT2
  assert(v.dt2_vec.size() == nv);
  assert(v.sumw_vec.size() == nv);
#endif
  assert(v.rho_vec.size() == nv);
  assert(v.swz_vec.size() == nv);
  assert(v.swt_vec.size() == nv);
  assert(v.exp_arg_vec.size() == nv);
  assert(v.exp_vec.size() == nv);
  assert(v.se_vec.size() == nv);
  assert(v.nuz_vec.size() == nv);
  assert(v.nut_vec.size() == nv);
  assert(v.szz_vec.size() == nv);
  assert(v.stt_vec.size() == nv);
  assert(v.szt_vec.size() == nv);

  assert(v.zvtx == &v.zvtx_vec.front());
  assert(v.tvtx == &v.tvtx_vec.front());
  assert(v.rho == &v.rho_vec.front());
  assert(v.exp_arg == &v.exp_arg_vec.front());
  assert(v.swz == &v.swz_vec.front());
  assert(v.swt == &v.swt_vec.front());
  assert(v.se == &v.se_vec.front());
  assert(v.nuz == &v.nuz_vec.front());
  assert(v.nut == &v.nut_vec.front());
  assert(v.szz == &v.szz_vec.front());
  assert(v.stt == &v.stt_vec.front());
  assert(v.szt == &v.szt_vec.front());
#ifdef USEVTXDT2
  assert(v.sumw == &v.sumw_vec.front());
  assert(v.dt2 == &v.dt2_vec.front());
#endif

  for (unsigned int k = 0; k < nv - 1; k++) {
    if (v.zvtx[k] <= v.zvtx[k + 1])
      continue;
    cout << " ZT, cluster z-ordering assertion failure   z[" << k << "] =" << v.zvtx[k] << "    z[" << k + 1
         << "] =" << v.zvtx[k + 1] << endl;
  }
  //for(unsigned int k=0; k< nv-1; k++){
  //assert( v.z[k] <= v.z[k+1]);
  //}

  // tracks
  assert(nt == tks.zpca_vec.size());
  assert(nt == tks.tpca_vec.size());
  assert(nt == tks.dz2_vec.size());
  assert(nt == tks.dt2_vec.size());
  assert(nt == tks.tt.size());
  assert(nt == tks.tkwt_vec.size());
  assert(nt == tks.sum_Z_vec.size());
  assert(nt == tks.kmin.size());
  assert(nt == tks.kmax.size());

  assert(tks.zpca == &tks.zpca_vec.front());
  assert(tks.tpca == &tks.tpca_vec.front());
  assert(tks.dz2 == &tks.dz2_vec.front());
  assert(tks.dt2 == &tks.dt2_vec.front());
  assert(tks.tkwt == &tks.tkwt_vec.front());
  assert(tks.sum_Z == &tks.sum_Z_vec.front());

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
DAClusterizerInZT_vect::track_t DAClusterizerInZT_vect::fill(const vector<reco::TransientTrack>& tracks) const {
  // prepare track data for clustering
  track_t tks;
  double sumtkwt = 0.;

  for (const auto& tk : tracks) {
    if (!tk.isValid())
      continue;
    double t_tkwt = 1.;
    double t_z = tk.stateAtBeamLine().trackStateAtPCA().position().z();
    double t_t = tk.timeExt();

    if (std::fabs(t_z) > 1000.)
      continue;
    /*  for comparison with 1d clustering, keep such tracks without timing info, see below
    if (std::abs(t_t) > t0Max_)
	continue;
    */

    auto const& t_mom = tk.stateAtBeamLine().trackStateAtPCA().momentum();
    //  get the beam-spot
    reco::BeamSpot beamspot = tk.stateAtBeamLine().beamSpot();
    double t_dz2 = std::pow(tk.track().dzError(), 2)  // track errror
                   + (std::pow(beamspot.BeamWidthX() * t_mom.x(), 2) + std::pow(beamspot.BeamWidthY() * t_mom.y(), 2)) *
                         std::pow(t_mom.z(), 2) / std::pow(t_mom.perp2(), 2)  // beam spot width
                   + std::pow(vertexSize_, 2);  // intrinsic vertex size, safer for outliers and short lived decays
    t_dz2 = 1. / t_dz2;
    if (edm::isNotFinite(t_dz2) || t_dz2 < std::numeric_limits<double>::min()) {
      edm::LogWarning("DAClusterizerinZT_vect") << "rejected track t_dz2 " << t_dz2;
      continue;
    }

    double t_dt2 =
        std::pow(tk.dtErrorExt(), 2.) +
        std::pow(vertexSizeTime_, 2.);  // the ~injected~ timing error, need to add a small minimum vertex size in time
    if ((tk.dtErrorExt() > TransientTrackBuilder::defaultInvalidTrackTimeReso) || (std::abs(t_t) > t0Max_)) {
      t_dt2 = 0;  // tracks with no time measurement
    } else {
      t_dt2 = 1. / t_dt2;
      if (edm::isNotFinite(t_dt2) || t_dt2 < std::numeric_limits<double>::min()) {
        edm::LogWarning("DAClusterizerinZT_vect") << "rejected track t_dt2 " << t_dt2;
        continue;
      }
    }

    if (d0CutOff_ > 0) {
      Measurement1D atIP = tk.stateAtBeamLine().transverseImpactParameter();  // error contains beamspot
      t_tkwt = 1. / (1. + local_exp(std::pow(atIP.value() / atIP.error(), 2) -
                                    std::pow(d0CutOff_, 2)));  // reduce weight for high ip tracks
      if (edm::isNotFinite(t_tkwt) || t_tkwt < std::numeric_limits<double>::epsilon()) {
        edm::LogWarning("DAClusterizerinZT_vect") << "rejected track t_tkwt " << t_tkwt;
        continue;  // usually is > 0.99
      }
    }

    tks.addItem(t_z, t_t, t_dz2, t_dt2, &tk, t_tkwt);
    sumtkwt += t_tkwt;
  }

  tks.extractRaw();
  tks.osumtkwt = sumtkwt > 0 ? 1. / sumtkwt : 0.;

#ifdef DEBUG
  if (DEBUGLEVEL > 0) {
    std::cout << "Track count (ZT) filled " << tks.getSize() << " initial " << tracks.size() << std::endl;
  }
#endif

  return tks;
}

namespace {
  inline double Eik(double t_z, double k_z, double t_dz2, double t_t, double k_t, double t_dt2) {
    return std::pow(t_z - k_z, 2) * t_dz2 + std::pow(t_t - k_t, 2) * t_dt2;
  }
}  // namespace

void DAClusterizerInZT_vect::set_vtx_range(double beta, track_t& gtracks, vertex_t& gvertices) const {
  const unsigned int nv = gvertices.getSize();
  const unsigned int nt = gtracks.getSize();

  if (nv == 0) {
    edm::LogWarning("DAClusterizerinZT_vect") << "empty cluster list in set_vtx_range";
    return;
  }

  for (auto itrack = 0U; itrack < nt; ++itrack) {
    double zrange = max(sel_zrange_ / sqrt(beta * gtracks.dz2[itrack]), zrange_min_);

    double zmin = gtracks.zpca[itrack] - zrange;
    unsigned int kmin = min(nv - 1, gtracks.kmin[itrack]);
    // find the smallest vertex_z that is larger than zmin
    if (gvertices.zvtx[kmin] > zmin) {
      while ((kmin > 0) && (gvertices.zvtx[kmin - 1] > zmin)) {
        kmin--;
      }
    } else {
      while ((kmin < (nv - 1)) && (gvertices.zvtx[kmin] < zmin)) {
        kmin++;
      }
    }

    double zmax = gtracks.zpca[itrack] + zrange;
    unsigned int kmax = min(nv - 1, gtracks.kmax[itrack] - 1);
    // note: kmax points to the last vertex in the range, while gtracks.kmax points to the entry BEHIND the last vertex
    // find the largest vertex_z that is smaller than zmax
    if (gvertices.zvtx[kmax] < zmax) {
      while ((kmax < (nv - 1)) && (gvertices.zvtx[kmax + 1] < zmax)) {
        kmax++;
      }
    } else {
      while ((kmax > 0) && (gvertices.zvtx[kmax] > zmax)) {
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

#ifdef DEBUG
    if (gtracks.kmin[itrack] >= gtracks.kmax[itrack]) {
      cout << "set_vtx_range trk = " << itrack << "  kmin,kmax=" << kmin << "," << kmax
           << " gtrack.kmin,kmax = " << gtracks.kmin[itrack] << "," << gtracks.kmax[itrack] << " zrange = " << zrange
           << endl;
    }
#endif
  }
}

void DAClusterizerInZT_vect::clear_vtx_range(track_t& gtracks, vertex_t& gvertices) const {
  const unsigned int nt = gtracks.getSize();
  const unsigned int nv = gvertices.getSize();
  for (auto itrack = 0U; itrack < nt; ++itrack) {
    gtracks.kmin[itrack] = 0;
    gtracks.kmax[itrack] = nv;
  }
}

double DAClusterizerInZT_vect::update(
    double beta, track_t& gtracks, vertex_t& gvertices, const double rho0, const bool updateTc) const {
  //update weights and vertex positions
  // returns the maximum of changes of vertex positions
  // sums needed for Tc are only updated if updateTC == true

  const unsigned int nt = gtracks.getSize();
  const unsigned int nv = gvertices.getSize();
  auto osumtkwt = gtracks.osumtkwt;

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
    const auto track_z = tracks.zpca[itrack];
    const auto track_t = tracks.tpca[itrack];
    const auto botrack_dz2 = -beta * tracks.dz2[itrack];
#ifndef USEVTXDT2
    const auto botrack_dt2 = -beta * tracks.dt2[itrack];
#endif
    // auto-vectorized
    for (unsigned int ivertex = kmin; ivertex < kmax; ++ivertex) {
      const auto mult_resz = track_z - vertices.zvtx[ivertex];
      const auto mult_rest = track_t - vertices.tvtx[ivertex];
#ifdef USEVTXDT2
      const auto botrack_dt2 =
          -beta * tracks.dt2[itrack] * vertices.dt2[ivertex] / (tracks.dt2[itrack] + vertices.dt2[ivertex] + 1.e-10);
#endif
      vertices.exp_arg[ivertex] = botrack_dz2 * (mult_resz * mult_resz) + botrack_dt2 * (mult_rest * mult_rest);
    }
  };

  auto kernel_add_Z_range = [Z_init](
                                vertex_t const& vertices, const unsigned int kmin, const unsigned int kmax) -> double {
    double ZTemp = Z_init;
    for (unsigned int ivertex = kmin; ivertex < kmax; ++ivertex) {
      ZTemp += vertices.rho[ivertex] * vertices.exp[ivertex];
    }
    return ZTemp;
  };

  auto kernel_calc_normalization_range = [updateTc](const unsigned int track_num,
                                                    track_t& tracks,
                                                    vertex_t& vertices,
                                                    const unsigned int kmin,
                                                    const unsigned int kmax) {
    auto tmp_trk_tkwt = tracks.tkwt[track_num];
    auto o_trk_sum_Z = 1. / tracks.sum_Z[track_num];
    auto o_trk_err_z = tracks.dz2[track_num];
    auto o_trk_err_t = tracks.dt2[track_num];
    auto tmp_trk_z = tracks.zpca[track_num];
    auto tmp_trk_t = tracks.tpca[track_num];
    // auto-vectorized
    if (updateTc) {
#pragma GCC ivdep
      for (unsigned int k = kmin; k < kmax; ++k) {
        // parens are important for numerical stability
        vertices.se[k] += tmp_trk_tkwt * (vertices.exp[k] * o_trk_sum_Z);
        const auto w = tmp_trk_tkwt * (vertices.rho[k] * vertices.exp[k] * o_trk_sum_Z);  // p_{ik}
        const auto wz = w * o_trk_err_z;
        const auto wt = w * o_trk_err_t;
#ifdef USEVTXDT2
        vertices.sumw[k] += w;  // for vtxdt2
#endif
        vertices.nuz[k] += wz;
        vertices.nut[k] += wt;
        vertices.swz[k] += wz * tmp_trk_z;
        vertices.swt[k] += wt * tmp_trk_t;
        // the following lines are only needed for Tc updates
        const auto dsz = (tmp_trk_z - vertices.zvtx[k]) * o_trk_err_z;
        const auto dst = (tmp_trk_t - vertices.tvtx[k]) * o_trk_err_t;
        vertices.szz[k] += w * dsz * dsz;
        vertices.stt[k] += w * dst * dst;
        vertices.szt[k] += w * dsz * dst;
      }
    } else {
#pragma GCC ivdep
      for (unsigned int k = kmin; k < kmax; ++k) {
        vertices.se[k] += tmp_trk_tkwt * (vertices.exp[k] * o_trk_sum_Z);
        const auto w = tmp_trk_tkwt * (vertices.rho[k] * vertices.exp[k] * o_trk_sum_Z);  // p_{ik}
        const auto wz = w * o_trk_err_z;
        const auto wt = w * o_trk_err_t;
#ifdef USEVTXDT2
        vertices.sumw[k] += w;  // for vtxdt2
#endif
        vertices.nuz[k] += wz;
        vertices.nut[k] += wt;
        vertices.swz[k] += wz * tmp_trk_z;
        vertices.swt[k] += wt * tmp_trk_t;
      }
    }
  };

  for (auto ivertex = 0U; ivertex < nv; ++ivertex) {
    gvertices.se[ivertex] = 0.0;
    gvertices.nuz[ivertex] = 0.0;
    gvertices.nut[ivertex] = 0.0;
    gvertices.swz[ivertex] = 0.0;
    gvertices.swt[ivertex] = 0.0;
    gvertices.szz[ivertex] = 0.0;
    gvertices.stt[ivertex] = 0.0;
    gvertices.szt[ivertex] = 0.0;
#ifdef USEVTXDT2
    gvertices.sumw[ivertex] = 0.0;
#endif
  }

  // loop over tracks
  for (auto itrack = 0U; itrack < nt; ++itrack) {
    const unsigned int kmin = gtracks.kmin[itrack];
    const unsigned int kmax = gtracks.kmax[itrack];

#ifdef DEBUG
    assert((kmin < kmax) && (kmax <= nv));
    assert(itrack < gtracks.sum_Z_vec.size());
#endif

    kernel_calc_exp_arg_range(itrack, gtracks, gvertices, kmin, kmax);
    local_exp_list_range(gvertices.exp_arg, gvertices.exp, kmin, kmax);
    gtracks.sum_Z[itrack] = kernel_add_Z_range(gvertices, kmin, kmax);

    if (edm::isNotFinite(gtracks.sum_Z[itrack]))
      gtracks.sum_Z[itrack] = 0.0;

    if (gtracks.sum_Z[itrack] > 1.e-100) {
      kernel_calc_normalization_range(itrack, gtracks, gvertices, kmin, kmax);
    }
  }

  // now update z, t, and rho
  auto kernel_calc_zt = [osumtkwt, nv](vertex_t& vertices) -> double {
    double delta = 0;

    // does not vectorize
    for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
      if (vertices.nuz[ivertex] > 0.) {
        auto znew = vertices.swz[ivertex] / vertices.nuz[ivertex];
        delta = max(std::abs(vertices.zvtx[ivertex] - znew), delta);
        vertices.zvtx[ivertex] = znew;
      }

      if (vertices.nut[ivertex] > 0.) {
        auto tnew = vertices.swt[ivertex] / vertices.nut[ivertex];
        //delta = max(std::abs(vertices.t[ ivertex ] - tnew), delta); // FIXME
        vertices.tvtx[ivertex] = tnew;
#ifdef USEVTXDT2
        vertices.dt2[ivertex] = vertices.nut[ivertex] / vertices.sumw[ivertex];
#endif
      } else {
        // FIXME
        // apparently this cluster has not timing info attached
        // this should be taken into account somehow, otherwise we might fail to attach
        // new tracks that do have timing information
        vertices.tvtx[ivertex] = 0;
#ifdef USEVTXDT2
        vertices.dt2[ivertex] = 0;
#endif
      }

#ifdef DEBUG
      if ((vertices.nut[ivertex] <= 0.) && (vertices.nuz[ivertex] <= 0.)) {
        edm::LogInfo("sumw") << "invalid sum of weights in fit: " << endl;
        std::cout << " a cluster melted away ?  "
                  << "  zk=" << vertices.zvtx[ivertex] << "  rho=" << vertices.rho[ivertex]
                  << " sumw(z,t) =" << vertices.nuz[ivertex] << "," << vertices.nut[ivertex] << endl;
        // FIXME: discard this cluster
      }
#endif
    }

    for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
      vertices.rho[ivertex] = vertices.rho[ivertex] * vertices.se[ivertex] * osumtkwt;
    }

    return delta;
  };

  double delta = kernel_calc_zt(gvertices);

  if (zorder(gvertices)) {
    set_vtx_range(beta, gtracks, gvertices);
  };

  // return how much the prototypes moved
  return delta;
}

bool DAClusterizerInZT_vect::zorder(vertex_t& y) const {
  const unsigned int nv = y.getSize();

#ifdef DEBUG
  assert(y.zvtx_vec.size() == nv);
  assert(y.tvtx_vec.size() == nv);
  assert(y.rho_vec.size() == nv);
#endif

  if (nv < 2)
    return false;

  bool reordering = true;
  bool modified = false;

  while (reordering) {
    reordering = false;
    for (unsigned int k = 0; (k + 1) < nv; k++) {
      if (y.zvtx[k + 1] < y.zvtx[k]) {
        auto ztemp = y.zvtx[k];
        y.zvtx[k] = y.zvtx[k + 1];
        y.zvtx[k + 1] = ztemp;
        auto ttemp = y.tvtx[k];
        y.tvtx[k] = y.tvtx[k + 1];
        y.tvtx[k + 1] = ttemp;
#ifdef USEVTXDT2
        auto dt2temp = y.dt2[k];
        y.dt2[k] = y.dt2[k + 1];
        y.dt2[k + 1] = dt2temp;
#endif
        auto ptemp = y.rho[k];
        y.rho[k] = y.rho[k + 1];
        y.rho[k + 1] = ptemp;
        reordering = true;
      }
    }
    modified |= reordering;
  }

  if (modified) {
    y.extractRaw();
    return true;
  }

  return false;
}

bool DAClusterizerInZT_vect::find_nearest(
    double z, double t, vertex_t& y, unsigned int& k_min, double dz, double dt) const {
  // find the cluster nearest to (z,t)
  // distance measure is   delta = (delta_z / dz )**2  + (delta_t/ d_t)**2
  // assumes that clusters are ordered n z
  // return value is false if no neighbour with distance < 1 is found

  unsigned int nv = y.getSize();
  if (nv < 2) {
    k_min = 0;
    return false;
  }

  // find nearest in z, binary search later
  unsigned int k = 0;
  for (unsigned int k0 = 1; k0 < nv; k0++) {
    if (std::abs(y.zvtx[k0] - z) < std::abs(y.zvtx[k] - z)) {
      k = k0;
    }
  }

  double delta_min = 1.;

  //search left
  unsigned int k1 = k;
  while ((k1 > 0) && ((y.zvtx[k] - y.zvtx[--k1]) < dz)) {
    auto delta = std::pow((y.zvtx[k] - y.zvtx[k1]) / dz, 2) + std::pow((y.tvtx[k] - y.tvtx[k1]) / dt, 2);
    if (delta < delta_min) {
      k_min = k1;
      delta_min = delta;
    }
  }

  //search right
  k1 = k;
  while (((++k1) < nv) && ((y.zvtx[k1] - y.zvtx[k]) < dz)) {
    auto delta = std::pow((y.zvtx[k1] - y.zvtx[k]) / dz, 2) + std::pow((y.tvtx[k1] - y.tvtx[k]) / dt, 2);
    if (delta < delta_min) {
      k_min = k1;
      delta_min = delta;
    }
  }

  return (delta_min < 1.);
}

unsigned int DAClusterizerInZT_vect::thermalize(
    double beta, track_t& tks, vertex_t& v, const double delta_max0, const double rho0) const {
  unsigned int niter = 0;
  double delta = 0;
  double delta_max = delta_lowT_;

  if (convergence_mode_ == 0) {
    delta_max = delta_max0;
  } else if (convergence_mode_ == 1) {
    delta_max = delta_lowT_ / sqrt(std::max(beta, 1.0));
  }

  zorder(v);
  set_vtx_range(beta, tks, v);

  double delta_sum_range = 0;  // accumulate max(|delta-z|) as a lower bound
  std::vector<double> z0 = v.zvtx_vec;

  while (niter++ < maxIterations_) {
    delta = update(beta, tks, v, rho0);
    delta_sum_range += delta;

    if (delta_sum_range > zrange_min_) {
      for (unsigned int k = 0; k < v.getSize(); k++) {
        if (std::abs(v.zvtx_vec[k] - z0[k]) > zrange_min_) {
          zorder(v);
          set_vtx_range(beta, tks, v);
          delta_sum_range = 0;
          z0 = v.zvtx_vec;
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
    std::cout << "DAClusterizerInZT_vect.thermalize niter = " << niter << " at T = " << 1 / beta
              << "  nv = " << v.getSize() << std::endl;
    if (DEBUGLEVEL > 2)
      dump(beta, v, tks, 0);
  }
#endif

  return niter;
}

bool DAClusterizerInZT_vect::merge(vertex_t& y, track_t& tks, double& beta) const {
  // merge clusters that collapsed or never separated,
  // return true if vertices were merged, false otherwise
  const unsigned int nv = y.getSize();

  if (nv < 2)
    return false;

  // merge the smallest distance clusters first
  unsigned int k1_min = 0, k2_min = 0;
  double delta_min = 0;

  for (unsigned int k1 = 0; (k1 + 1) < nv; k1++) {
    unsigned int k2 = k1;
    while ((++k2 < nv) && (std::fabs(y.zvtx[k2] - y.zvtx[k1]) < zmerge_)) {
      auto delta = std::pow((y.zvtx[k2] - y.zvtx[k1]) / zmerge_, 2) + std::pow((y.tvtx[k2] - y.tvtx[k1]) / tmerge_, 2);
      if ((delta < delta_min) || (k1_min == k2_min)) {
        k1_min = k1;
        k2_min = k2;
        delta_min = delta;
      }
    }
  }

  if ((k1_min == k2_min) || (delta_min > 1)) {
    return false;
  }

  double rho = y.rho[k1_min] + y.rho[k2_min];

#ifdef DEBUG
  assert((k1_min < nv) && (k2_min < nv));
  if (DEBUGLEVEL > 1) {
    std::cout << "merging (" << setw(8) << fixed << setprecision(4) << y.zvtx[k1_min] << ',' << y.tvtx[k1_min]
              << ") and (" << y.zvtx[k2_min] << ',' << y.tvtx[k2_min] << ")"
              << "  idx=" << k1_min << "," << k2_min << std::endl;
  }
#endif

  if (rho > 0) {
    y.zvtx[k1_min] = (y.rho[k1_min] * y.zvtx[k1_min] + y.rho[k2_min] * y.zvtx[k2_min]) / rho;
    y.tvtx[k1_min] = (y.rho[k1_min] * y.tvtx[k1_min] + y.rho[k2_min] * y.tvtx[k2_min]) / rho;
#ifdef USEVTXDT2
    y.dt2[k1_min] = (y.rho[k1_min] * y.dt2[k1_min] + y.rho[k2_min] * y.dt2[k2_min]) / rho;
#endif
  } else {
    y.zvtx[k1_min] = 0.5 * (y.zvtx[k1_min] + y.zvtx[k2_min]);
    y.tvtx[k1_min] = 0.5 * (y.tvtx[k1_min] + y.tvtx[k2_min]);
  }
  y.rho[k1_min] = rho;
  y.removeItem(k2_min, tks);

  zorder(y);
  set_vtx_range(beta, tks, y);
  y.extractRaw();
  return true;
}

bool DAClusterizerInZT_vect::purge(vertex_t& y, track_t& tks, double& rho0, const double beta) const {
  constexpr double eps = 1.e-100;
  // eliminate clusters with only one significant/unique track
  const unsigned int nv = y.getSize();
  const unsigned int nt = tks.getSize();

  if (nv < 2)
    return false;

  std::vector<double> sump_v(nv), arg_cache_v(nv), exp_cache_v(nv), pcut_cache_v(nv);
  std::vector<int> nUnique_v(nv);
  double* __restrict__ parg_cache;
  double* __restrict__ pexp_cache;
  double* __restrict__ ppcut_cache;
  double* __restrict__ psump;
  int* __restrict__ pnUnique;
  int constexpr nunique_min_ = 2;

  zorder(y);
  set_vtx_range(beta, tks, y);

  parg_cache = arg_cache_v.data();
  pexp_cache = exp_cache_v.data();
  ppcut_cache = pcut_cache_v.data();
  psump = sump_v.data();
  pnUnique = nUnique_v.data();

  const auto rhoconst = rho0 * local_exp(-beta * dzCutOff_ * dzCutOff_);
  for (unsigned int k = 0; k < nv; ++k) {
    const double pmax = y.rho[k] / (y.rho[k] + rhoconst);
    ppcut_cache[k] = uniquetrkweight_ * pmax;
  }

  for (unsigned int i = 0; i < nt; i++) {
    const auto invZ = ((tks.sum_Z[i] > eps) && (tks.tkwt[i] > uniquetrkminp_)) ? 1. / tks.sum_Z[i] : 0.;
    const auto track_z = tks.zpca[i];
    const auto track_t = tks.tpca[i];
    const auto botrack_dz2 = -beta * tks.dz2[i];
    const auto botrack_dt2 = -beta * tks.dt2[i];
    const auto kmin = tks.kmin[i];
    const auto kmax = tks.kmax[i];

    for (unsigned int k = kmin; k < kmax; k++) {
      const auto mult_resz = track_z - y.zvtx[k];
      const auto mult_rest = track_t - y.tvtx[k];
      parg_cache[k] = botrack_dz2 * (mult_resz * mult_resz) + botrack_dt2 * (mult_rest * mult_rest);
    }

    local_exp_list_range(parg_cache, pexp_cache, kmin, kmax);

    for (unsigned int k = kmin; k < kmax; k++) {
      const double p = y.rho[k] * pexp_cache[k] * invZ;
      psump[k] += p;
      pnUnique[k] += (p > ppcut_cache[k]) ? 1 : 0;
    }
  }

  double sumpmin = nt;
  unsigned int k0 = nv;
  for (unsigned k = 0; k < nv; k++) {
    if ((pnUnique[k] < nunique_min_) && (psump[k] < sumpmin)) {
      sumpmin = psump[k];
      k0 = k;
    }
  }

  if (k0 != nv) {
#ifdef DEBUG
    assert(k0 < y.getSize());
    if (DEBUGLEVEL > 1) {
      std::cout << "eliminating prototype at " << std::setw(10) << std::setprecision(4) << y.zvtx[k0] << ","
                << y.tvtx[k0] << " with sump=" << sumpmin << "  rho*nt =" << y.rho[k0] * nt << endl;
    }
#endif
    y.removeItem(k0, tks);
    set_vtx_range(beta, tks, y);
    return true;
  } else {
    return false;
  }
}

double DAClusterizerInZT_vect::beta0(double betamax, track_t const& tks, vertex_t const& y) const {
  double T0 = 0;  // max Tc for beta=0
  // estimate critical temperature from beta=0 (T=inf)
  const unsigned int nt = tks.getSize();
  const unsigned int nv = y.getSize();

  for (unsigned int k = 0; k < nv; k++) {
    // vertex fit at T=inf
    double sumwz = 0;
    double sumwt = 0;
    double sumw_z = 0;
    double sumw_t = 0;
    for (unsigned int i = 0; i < nt; i++) {
      double w_z = tks.tkwt[i] * tks.dz2[i];
      double w_t = tks.tkwt[i] * tks.dt2[i];
      sumwz += w_z * tks.zpca[i];
      sumwt += w_t * tks.tpca[i];
      sumw_z += w_z;
      sumw_t += w_t;
    }
    y.zvtx[k] = sumwz / sumw_z;
    y.tvtx[k] = sumwt / sumw_t;

    // estimate Tc, eventually do this in the same loop
    double szz = 0, stt = 0, szt = 0;
    double nuz = 0, nut = 0;
    for (unsigned int i = 0; i < nt; i++) {
      double dz = (tks.zpca[i] - y.zvtx[k]) * tks.dz2[i];
      double dt = (tks.tpca[i] - y.tvtx[k]) * tks.dt2[i];
      double w = tks.tkwt[i];
      szz += w * dz * dz;
      stt += w * dt * dt;
      szt += w * dz * dt;
      nuz += w * tks.dz2[i];
      nut += w * tks.dt2[i];
    }
    double Tz = szz / nuz;
    double Tt = 0;
    double Tc = 0;
    if (nut > 0) {
      Tt = stt / nut;
      Tc = Tz + Tt + sqrt(pow(Tz - Tt, 2) + 4 * szt * szt / nuz / nut);
    } else {
      Tc = 2. * Tz;
    }

    if (Tc > T0)
      T0 = Tc;
  }  // vertex loop (normally there should be only one vertex at beta=0)

#ifdef DEBUG
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZT_vect.beta0:   Tc = " << T0 << std::endl;
    int coolingsteps = 1 - int(std::log(T0 * betamax) / std::log(coolingFactor_));
    std::cout << "DAClusterizerInZT_vect.beta0:   nstep = " << coolingsteps << std::endl;
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

double DAClusterizerInZT_vect::get_Tc(const vertex_t& y, int k) const {
  double Tz = y.szz[k] / y.nuz[k];  // actually 0.5*Tc(z)
  double Tt = 0.;
  if (y.nut[k] > 0) {
    Tt = y.stt[k] / y.nut[k];
    double mx = y.szt[k] / y.nuz[k] * y.szt[k] / y.nut[k];
    return Tz + Tt + sqrt(pow(Tz - Tt, 2) + 4 * mx);
  }
  return 2 * Tz;
}

bool DAClusterizerInZT_vect::split(const double beta, track_t& tks, vertex_t& y, double threshold) const {
  // split only critical vertices (Tc >~ T=1/beta   <==>   beta*Tc>~1)
  // an update must have been made just before doing this (same beta, no merging)
  // returns true if at least one cluster was split

  constexpr double epsilonz = 1e-3;  // minimum split size z
  constexpr double epsilont = 1e-2;  // minimum split size t
  unsigned int nv = y.getSize();
  const double twoBeta = 2.0 * beta;

  // avoid left-right biases by splitting highest Tc first

  std::vector<std::pair<double, unsigned int> > critical;
  for (unsigned int k = 0; k < nv; k++) {
    double Tc = get_Tc(y, k);
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

    // split direction in the (z,t)-plane

    double Mzz = y.nuz[k] - twoBeta * y.szz[k];
    double Mtt = y.nut[k] - twoBeta * y.stt[k];
    double Mzt = -twoBeta * y.szt[k];
    const double twoMzt = 2.0 * Mzt;
    double D = sqrt(pow(Mtt - Mzz, 2) + twoMzt * twoMzt);
    double q1 = atan2(-Mtt + Mzz + D, -twoMzt);
    double l1 = 0.5 * (-Mzz - Mtt + D);
    double l2 = 0.5 * (-Mzz - Mtt - D);
    if ((std::abs(l1) < 1e-4) && (std::abs(l2) < 1e-4)) {
      edm::LogWarning("DAClusterizerInZT_vect") << "warning, bad eigenvalues!  idx=" << k << " z= " << y.zvtx[k]
                                                << " Mzz=" << Mzz << "   Mtt=" << Mtt << "  Mzt=" << Mzt << endl;
    }

    double qsplit = q1;
    double cq = cos(qsplit);
    double sq = sin(qsplit);
    if (cq < 0) {
      cq = -cq;
      sq = -sq;
    }  // prefer cq>0 to keep z-ordering

    // estimate subcluster positions and weight
    double p1 = 0, z1 = 0, t1 = 0, wz1 = 0, wt1 = 0;
    double p2 = 0, z2 = 0, t2 = 0, wz2 = 0, wt2 = 0;
    for (unsigned int i = 0; i < nt; ++i) {
      if (tks.sum_Z[i] > 1.e-100) {
        double lr = (tks.zpca[i] - y.zvtx[k]) * cq + (tks.tpca[i] - y.tvtx[k]) * sq;
        // winner-takes-all, usually overestimates splitting
        double tl = lr < 0 ? 1. : 0.;
        double tr = 1. - tl;

        // soften it, especially at low T
        double arg = lr * std::sqrt(beta * (cq * cq * tks.dz2[i] + sq * sq * tks.dt2[i]));
        if (std::abs(arg) < 20) {
          double t = local_exp(-arg);
          tl = t / (t + 1.);
          tr = 1 / (t + 1.);
        }

        double p = y.rho[k] * tks.tkwt[i] *
                   local_exp(-beta * Eik(tks.zpca[i], y.zvtx[k], tks.dz2[i], tks.tpca[i], y.tvtx[k], tks.dt2[i])) /
                   tks.sum_Z[i];
        double wz = p * tks.dz2[i];
        double wt = p * tks.dt2[i];
        p1 += p * tl;
        z1 += wz * tl * tks.zpca[i];
        t1 += wt * tl * tks.tpca[i];
        wz1 += wz * tl;
        wt1 += wt * tl;
        p2 += p * tr;
        z2 += wz * tr * tks.zpca[i];
        t2 += wt * tr * tks.tpca[i];
        wz2 += wz * tr;
        wt2 += wt * tr;
      }
    }

    if (wz1 > 0) {
      z1 /= wz1;
    } else {
      z1 = y.zvtx[k] - epsilonz * cq;
      edm::LogWarning("DAClusterizerInZT_vect") << "warning, wz1 = " << scientific << wz1 << endl;
    }
    if (wt1 > 0) {
      t1 /= wt1;
    } else {
      t1 = y.tvtx[k] - epsilont * sq;
      edm::LogWarning("DAClusterizerInZT_vect") << "warning, wt1 = " << scientific << wt1 << endl;
    }
    if (wz2 > 0) {
      z2 /= wz2;
    } else {
      z2 = y.zvtx[k] + epsilonz * cq;
      edm::LogWarning("DAClusterizerInZT_vect") << "warning, wz2 = " << scientific << wz2 << endl;
    }
    if (wt2 > 0) {
      t2 /= wt2;
    } else {
      t2 = y.tvtx[k] + epsilont * sq;
      edm::LogWarning("DAClusterizerInZT_vect") << "warning, wt2 = " << scientific << wt2 << endl;
    }

    unsigned int k_min1 = k, k_min2 = k;
    constexpr double spliteps = 1e-8;
    while (((find_nearest(z1, t1, y, k_min1, epsilonz, epsilont) && (k_min1 != k)) ||
            (find_nearest(z2, t2, y, k_min2, epsilonz, epsilont) && (k_min2 != k))) &&
           (std::abs(z2 - z1) > spliteps || std::abs(t2 - t1) > spliteps)) {
      z1 = 0.5 * (z1 + y.zvtx[k]);
      t1 = 0.5 * (t1 + y.tvtx[k]);
      z2 = 0.5 * (z2 + y.zvtx[k]);
      t2 = 0.5 * (t2 + y.tvtx[k]);
    }

#ifdef DEBUG
    assert(k < nv);
    if (DEBUGLEVEL > 1) {
      if (std::fabs(y.zvtx[k] - zdumpcenter_) < zdumpwidth_) {
        std::cout << " T= " << std::setw(10) << std::setprecision(1) << 1. / beta << " Tc= " << critical[ic].first
                  << " direction =" << std::setprecision(4) << qsplit << "    splitting (" << std::setw(8) << std::fixed
                  << std::setprecision(4) << y.zvtx[k] << "," << y.tvtx[k] << ")"
                  << " --> (" << z1 << ',' << t1 << "),(" << z2 << ',' << t2 << ")     [" << p1 << "," << p2 << "]";
        if (std::fabs(z2 - z1) > epsilonz || std::fabs(t2 - t1) > epsilont) {
          std::cout << std::endl;
        } else {
          std::cout << "  rejected " << std::endl;
        }
      }
    }
#endif

    if (z1 > z2) {
      edm::LogInfo("DAClusterizerInZT") << "warning   swapping z in split  qsplit=" << qsplit << "   cq=" << cq
                                        << "  sq=" << sq << endl;
      auto ztemp = z1;
      auto ttemp = t1;
      auto ptemp = p1;
      z1 = z2;
      t1 = t2;
      p1 = p2;
      z2 = ztemp;
      t2 = ttemp;
      p2 = ptemp;
    }

    // split if the new subclusters are significantly separated
    if (std::fabs(z2 - z1) > epsilonz || std::fabs(t2 - t1) > epsilont) {
      split = true;
      double rho1 = p1 * y.rho[k] / (p1 + p2);
      double rho2 = p2 * y.rho[k] / (p1 + p2);

      // replace the original by (z2,t2)
      y.removeItem(k, tks);
      unsigned int k2 = y.insertOrdered(z2, t2, rho2, tks);

#ifdef DEBUG
      if (k2 < k) {
        std::cout << "unexpected z-ordering in split" << std::endl;
      }
#endif

      // adjust pointers if necessary
      if (!(k == k2)) {
        for (unsigned int jc = ic; jc < critical.size(); jc++) {
          if (critical[jc].second > k) {
            critical[jc].second--;
          }
          if (critical[jc].second >= k2) {
            critical[jc].second++;
          }
        }
      }

      // insert (z1,t1) where it belongs
      unsigned int k1 = y.insertOrdered(z1, t1, rho1, tks);
      nv++;

      // adjust remaining pointers
      for (unsigned int jc = ic; jc < critical.size(); jc++) {
        if (critical[jc].second >= k1) {
          critical[jc].second++;
        }
      }
    } else {
#ifdef DEBUG
      std::cout << "warning ! split rejected, too small." << endl;
#endif
    }
  }

  return split;
}

vector<TransientVertex> DAClusterizerInZT_vect::vertices(const vector<reco::TransientTrack>& tracks) const {
  track_t&& tks = fill(tracks);
  tks.extractRaw();

  unsigned int nt = tks.getSize();
  double rho0 = 0.0;  // start with no outlier rejection

  vector<TransientVertex> clusters;
  if (tks.getSize() == 0)
    return clusters;

  vertex_t y;  // the vertex prototypes

  // initialize:single vertex at infinite temperature
  y.addItem(0, 0, 1.0);
  clear_vtx_range(tks, y);

  // estimate first critical temperature
  double beta = beta0(betamax_, tks, y);
#ifdef DEBUG
  if (DEBUGLEVEL > 0)
    std::cout << "Beta0 is " << beta << std::endl;
#endif

  thermalize(beta, tks, y, delta_highT_, 0.);

  // annealing loop, stop when T<minT  (i.e. beta>1/minT)

  double betafreeze = betamax_ * sqrt(coolingFactor_);

  while (beta < betafreeze) {
    update(beta, tks, y, rho0, true);
    while (merge(y, tks, beta)) {
      if (zorder(y))
        set_vtx_range(beta, tks, y);
      update(beta, tks, y, rho0, true);
    }
    split(beta, tks, y);

    beta = beta / coolingFactor_;
    thermalize(beta, tks, y, delta_highT_, 0.);
  }

#ifdef DEBUG
  verify(y, tks);

  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZT_vect::vertices :"
              << "merging at T=" << 1 / beta << std::endl;
  }
#endif

  zorder(y);
  set_vtx_range(beta, tks, y);
  // ZT_vect version of merge() does not use Tc, but split() does
  update(beta, tks, y, rho0, true);

  while (merge(y, tks, beta)) {
    zorder(y);
    set_vtx_range(beta, tks, y);
    update(beta, tks, y, rho0, true);
  }

#ifdef DEBUG
  verify(y, tks);
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZT_vect::vertices :"
              << "splitting/merging at T=" << 1 / beta << std::endl;
  }
#endif

  unsigned int ntry = 0;
  double threshold = 1.0;
  while (split(beta, tks, y, threshold) && (ntry++ < 10)) {
    thermalize(beta, tks, y, delta_highT_, 0.);

    while (merge(y, tks, beta)) {
      if (zorder(y))
        set_vtx_range(beta, tks, y);
      update(beta, tks, y, rho0, true);  // updateTc in case we go back to split
    }

#ifdef DEBUG
    verify(y, tks);
#endif

    // relax splitting a bit to reduce multiple split-merge cycles of the same cluster
    threshold *= 1.1;
  }

#ifdef DEBUG
  verify(y, tks);
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZT_vect::vertices :"
              << "turning on outlier rejection at T=" << 1 / beta << std::endl;
  }
#endif

  // switch on outlier rejection at T=minT
  if (dzCutOff_ > 0) {
    rho0 = 1. / nt;
    for (unsigned int a = 0; a < 5; a++) {
      update(beta, tks, y, a * rho0 / 5);  // adiabatic turn-on
    }
  }

  thermalize(beta, tks, y, delta_lowT_, rho0);

#ifdef DEBUG
  verify(y, tks);
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZT_vect::vertices :"
              << "merging with outlier rejection at T=" << 1 / beta << std::endl;
  }
  if (DEBUGLEVEL > 2)
    dump(beta, y, tks, 2);
#endif

  // merge again  (some cluster split by outliers collapse here)
  zorder(y);
  while (merge(y, tks, beta)) {
    set_vtx_range(beta, tks, y);
    update(beta, tks, y, rho0);
  }

#ifdef DEBUG
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZT_vect::vertices :"
              << "after merging with outlier rejection at T=" << 1 / beta << std::endl;
  }
  if (DEBUGLEVEL > 1)
    dump(beta, y, tks, 2);
#endif

  // go down to the purging temperature (if it is lower than tmin)
  while (beta < betapurge_) {
    beta = min(beta / coolingFactor_, betapurge_);
    thermalize(beta, tks, y, delta_lowT_, rho0);
  }

#ifdef DEBUG
  verify(y, tks);
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZT_vect::vertices :"
              << "purging at T=" << 1 / beta << std::endl;
  }
#endif

  // eliminate insignificant vertices, this is more restrictive at higher T
  while (purge(y, tks, rho0, beta)) {
    thermalize(beta, tks, y, delta_lowT_, rho0);
    if (zorder(y)) {
      set_vtx_range(beta, tks, y);
    };
  }

#ifdef DEBUG
  verify(y, tks);
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZT_vect::vertices :"
              << "last cooling T=" << 1 / beta << std::endl;
  }
#endif

  // optionally cool some more without doing anything, to make the assignment harder
  while (beta < betastop_) {
    beta = min(beta / coolingFactor_, betastop_);
    thermalize(beta, tks, y, delta_lowT_, rho0);
    if (zorder(y)) {
      set_vtx_range(beta, tks, y);
    };
  }

#ifdef DEBUG
  verify(y, tks);
  if (DEBUGLEVEL > 0) {
    std::cout << "DAClusterizerInZT_vect::vertices :"
              << "stop cooling at T=" << 1 / beta << std::endl;
  }
  if (DEBUGLEVEL > 2)
    dump(beta, y, tks, 2);
#endif

  // new, merge here and not in "clusterize"
  // final merging step
  double betadummy = 1;
  while (merge(y, tks, betadummy))
    ;

  // select significant tracks and use a TransientVertex as a container
  zorder(y);
  set_vtx_range(beta, tks, y);
  const unsigned int nv = y.getSize();
  for (unsigned int k = 0; k < nv; k++)
    if (edm::isNotFinite(y.rho[k]) || edm::isNotFinite(y.zvtx[k])) {
      y.rho[k] = 0;
      y.zvtx[k] = 0;
    }

  const auto z_sum_init = rho0 * local_exp(-beta * dzCutOff_ * dzCutOff_);
  std::vector<std::vector<unsigned int> > vtx_track_indices(nv);
  for (unsigned int i = 0; i < nt; i++) {
    const auto kmin = tks.kmin[i];
    const auto kmax = tks.kmax[i];
    for (auto k = kmin; k < kmax; k++) {
      y.exp_arg[k] = -beta * Eik(tks.zpca[i], y.zvtx[k], tks.dz2[i], tks.tpca[i], y.tvtx[k], tks.dt2[i]);
    }

    local_exp_list_range(y.exp_arg, y.exp, kmin, kmax);

    tks.sum_Z[i] = z_sum_init;
    for (auto k = kmin; k < kmax; k++) {
      tks.sum_Z[i] += y.rho[k] * y.exp[k];
    }
    const double invZ = tks.sum_Z[i] > 1e-100 ? 1. / tks.sum_Z[i] : 0.0;

    for (auto k = kmin; k < kmax; k++) {
      double p = y.rho[k] * y.exp[k] * invZ;
      if (p > mintrkweight_) {
        // assign  track i -> vertex k (hard, mintrkweight_ should be >= 0.5 here
        vtx_track_indices[k].push_back(i);
        break;
      }
    }

  }  // track loop

  GlobalError dummyError(0.01, 0, 0.01, 0., 0., 0.01);
  for (unsigned int k = 0; k < nv; k++) {
    if (!vtx_track_indices[k].empty()) {
      GlobalPoint pos(0, 0, y.zvtx[k]);
      vector<reco::TransientTrack> vertexTracks;
      for (auto i : vtx_track_indices[k]) {
        vertexTracks.push_back(*(tks.tt[i]));
      }
      TransientVertex v(pos, dummyError, vertexTracks, 0);
      clusters.push_back(v);
    }
  }

  return clusters;
}

vector<vector<reco::TransientTrack> > DAClusterizerInZT_vect::clusterize(
    const vector<reco::TransientTrack>& tracks) const {
  vector<vector<reco::TransientTrack> > clusters;
  vector<TransientVertex>&& pv = vertices(tracks);

#ifdef DEBUG
  if (DEBUGLEVEL > 0) {
    std::cout << "###################################################" << endl;
    std::cout << "# vectorized DAClusterizerInZT_vect::clusterize   nt=" << tracks.size() << endl;
    std::cout << "# DAClusterizerInZT_vect::clusterize   pv.size=" << pv.size() << endl;
    std::cout << "###################################################" << endl;
  }
#endif

  if (pv.empty()) {
    return clusters;
  }

  // fill into clusters, don't merge
  for (auto k = pv.begin(); k != pv.end(); k++) {
    vector<reco::TransientTrack> aCluster = k->originalTracks();
    if (aCluster.size() > 1) {
      clusters.push_back(aCluster);
    }
  }

  return clusters;
}

void DAClusterizerInZT_vect::dump(const double beta, const vertex_t& y, const track_t& tks, int verbosity) const {
#ifdef DEBUG
  const unsigned int nv = y.getSize();
  const unsigned int nt = tks.getSize();

  std::vector<unsigned int> iz;
  for (unsigned int j = 0; j < nt; j++) {
    iz.push_back(j);
  }
  std::sort(iz.begin(), iz.end(), [tks](unsigned int a, unsigned int b) { return tks.zpca[a] < tks.zpca[b]; });
  std::cout << std::endl;
  std::cout << "-----DAClusterizerInZT::dump ----" << nv << "  clusters " << std::endl;
  string h = "                                                                                 ";
  std::cout << h << " k= ";
  for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
    if (std::fabs(y.zvtx[ivertex] - zdumpcenter_) < zdumpwidth_) {
      std::cout << setw(8) << fixed << ivertex;
    }
  }
  std::cout << endl;

  std::cout << h << " z= ";
  std::cout << setprecision(4);
  for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
    if (std::fabs(y.zvtx[ivertex] - zdumpcenter_) < zdumpwidth_) {
      std::cout << setw(8) << fixed << y.zvtx[ivertex];
    }
  }
  std::cout << endl;

  std::cout << h << " t= ";
  std::cout << setprecision(4);
  for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
    if (std::fabs(y.zvtx[ivertex] - zdumpcenter_) < zdumpwidth_) {
      std::cout << setw(8) << fixed << y.tvtx[ivertex];
    }
  }
  std::cout << endl;

  std::cout << "T=" << setw(15) << 1. / beta << " Tmin =" << setw(10) << 1. / betamax_
            << "                                               Tc= ";
  for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
    if (std::fabs(y.zvtx[ivertex] - zdumpcenter_) < zdumpwidth_) {
      double Tc = get_Tc(y, ivertex);
      std::cout << setw(8) << fixed << setprecision(1) << Tc;
    }
  }
  std::cout << endl;

  std::cout << h << "rho= ";
  double sumrho = 0;
  for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
    sumrho += y.rho[ivertex];
    if (std::fabs(y.zvtx[ivertex] - zdumpcenter_) > zdumpwidth_)
      continue;
    std::cout << setw(8) << setprecision(4) << fixed << y.rho[ivertex];
  }
  std::cout << endl;

  std::cout << h << "nt= ";
  for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
    sumrho += y.rho[ivertex];
    if (std::fabs(y.zvtx[ivertex] - zdumpcenter_) > zdumpwidth_)
      continue;
    std::cout << setw(8) << setprecision(1) << fixed << y.rho[ivertex] * nt;
  }
  std::cout << endl;

  if (verbosity > 0) {
    double E = 0, F = 0;
    std::cout << endl;
    std::cout << "----        z +/- dz          t +/- dt                ip +/-dip       pt    phi  eta   weights  ----"
              << endl;
    std::cout << setprecision(4);
    for (unsigned int i0 = 0; i0 < nt; i0++) {
      unsigned int i = iz[i0];
      if (tks.sum_Z[i] > 0) {
        F -= std::log(tks.sum_Z[i]) / beta;
      }
      double tz = tks.zpca[i];

      if (std::fabs(tz - zdumpcenter_) > zdumpwidth_)
        continue;
      std::cout << setw(4) << i << ")" << setw(8) << fixed << setprecision(4) << tz << " +/-" << setw(6)
                << sqrt(1. / tks.dz2[i]);

      if (tks.dt2[i] > 0) {
        std::cout << setw(8) << fixed << setprecision(4) << tks.tpca[i] << " +/-" << setw(6) << sqrt(1. / tks.dt2[i]);
      } else {
        std::cout << "                  ";
      }

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

      double sump = 0.;
      for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
        if (std::fabs(y.zvtx[ivertex] - zdumpcenter_) > zdumpwidth_)
          continue;

        if ((tks.tkwt[i] > 0) && (tks.sum_Z[i] > 0)) {
          double p =
              y.rho[ivertex] *
              local_exp(-beta *
                        Eik(tks.zpca[i], y.zvtx[ivertex], tks.dz2[i], tks.tpca[i], y.tvtx[ivertex], tks.dt2[i])) /
              tks.sum_Z[i];

          if ((ivertex >= tks.kmin[i]) && (ivertex < tks.kmax[i])) {
            if (p > 0.0001) {
              std::cout << setw(8) << setprecision(3) << p;
            } else {
              std::cout << "    _   ";  // tiny but in the cluster list
            }
            E += p * Eik(tks.zpca[i], y.zvtx[ivertex], tks.dz2[i], tks.tpca[i], y.tvtx[ivertex], tks.dt2[i]);
            sump += p;
          } else {
            if (p > 0.1) {
              std::cout << "XXXXXXXX";  // we have an inconsistency here
            } else if (p > 0.0001) {
              std::cout << "X" << setw(6) << setprecision(3) << p << "X";
            } else {
              std::cout << "    .   ";  // not in the cluster list
            }
          }
        } else {
          std::cout << "        ";
        }
      }
      std::cout << "  ( " << std::setw(3) << tks.kmin[i] << "," << std::setw(3) << tks.kmax[i] - 1 << " ) ";
      std::cout << endl;
    }
    std::cout << "                                                                                    ";
    for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
      if (std::fabs(y.zvtx[ivertex] - zdumpcenter_) < zdumpwidth_) {
        std::cout << "   " << setw(3) << ivertex << "  ";
      }
    }
    std::cout << endl;
    std::cout << "                                                                                 z= ";
    std::cout << setprecision(4);
    for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
      if (std::fabs(y.zvtx[ivertex] - zdumpcenter_) < zdumpwidth_) {
        std::cout << setw(8) << fixed << y.zvtx[ivertex];
      }
    }
    std::cout << endl;
    std::cout << endl
              << "T=" << 1 / beta << " E=" << E << " n=" << y.getSize() << "  F= " << F << endl
              << "----------" << endl;
  }
#endif
}

void DAClusterizerInZT_vect::fillPSetDescription(edm::ParameterSetDescription& desc) {
  DAClusterizerInZ_vect::fillPSetDescription(desc);
  desc.add<double>("tmerge", 0.1);            // 4D only
  desc.add<double>("dtCutOff", 4.);           // 4D only
  desc.add<double>("t0Max", 1.0);             // 4D only
  desc.add<double>("vertexSizeTime", 0.008);  // 4D only
}
