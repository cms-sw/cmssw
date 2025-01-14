#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <stdexcept>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/SimpleSeedingLayersTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/FitUtils.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "Geometry/CommonTopologies/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/FitUtils.h"
#include "storeTracks.h"

/**
 * This class creates "legacy" reco::Track
 * objects from the output of SoA CA.
 */

//#define GPU_DEBUG

template <typename TrackerTraits>
class PixelTrackProducerFromSoAAlpaka : public edm::global::EDProducer<> {
  using TrackSoAHost = TracksHost<TrackerTraits>;
  using TracksHelpers = TracksUtilities<TrackerTraits>;
  using HMSstorage = std::vector<uint32_t>;
  using IndToEdm = std::vector<uint32_t>;

public:
  explicit PixelTrackProducerFromSoAAlpaka(const edm::ParameterSet &iConfig);
  ~PixelTrackProducerFromSoAAlpaka() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const override;

  // Event Data tokens
  const edm::EDGetTokenT<reco::BeamSpot> tBeamSpot_;

  const edm::EDGetTokenT<TrackSoAHost> tokenTrack_;
  const edm::EDGetTokenT<SiPixelRecHitCollectionNew> cpuPixelHits_;
  edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> cpuStripHits_;
  edm::EDGetTokenT<HMSstorage> hmsToken_;

  // Event Setup tokens
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> idealMagneticFieldToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttTopoToken_;

  int32_t const minNumberOfHits_;
  pixelTrack::Quality const minQuality_;
  const bool useStripHits_;
};

template <typename TrackerTraits>
PixelTrackProducerFromSoAAlpaka<TrackerTraits>::PixelTrackProducerFromSoAAlpaka(const edm::ParameterSet &iConfig)
    : tBeamSpot_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      tokenTrack_(consumes(iConfig.getParameter<edm::InputTag>("trackSrc"))),
      cpuPixelHits_(consumes<SiPixelRecHitCollectionNew>(iConfig.getParameter<edm::InputTag>("pixelRecHitLegacySrc"))),
      // hmsToken_(consumes<HMSstorage>(iConfig.getParameter<edm::InputTag>("pixelRecHitLegacySrc"))),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      idealMagneticFieldToken_(esConsumes()),
      ttTopoToken_(esConsumes()),
      minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits")),
      minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
      useStripHits_(iConfig.getParameter<bool>("useStripHits")) {
  if (minQuality_ == pixelTrack::Quality::notQuality) {
    throw cms::Exception("PixelTrackConfiguration")
        << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
  }
  if (minQuality_ < pixelTrack::Quality::bad) {
    throw cms::Exception("PixelTrackConfiguration")
        << iConfig.getParameter<std::string>("minQuality") + " not supported";
  }
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  // TrackCollection refers to TrackingRechit and TrackExtra
  // collections, need to declare its production after them to work
  // around a rare race condition in framework scheduling
  produces<reco::TrackCollection>();
  produces<IndToEdm>();
  if (useStripHits_) {
    cpuStripHits_ =
        consumes<SiStripMatchedRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stripRecHitLegacySrc"));
    hmsToken_ = consumes<HMSstorage>(iConfig.getParameter<edm::InputTag>("hitModuleStartSrc"));
  } else {
    hmsToken_ = consumes<HMSstorage>(iConfig.getParameter<edm::InputTag>("pixelRecHitLegacySrc"));
  }
}

template <typename TrackerTraits>
void PixelTrackProducerFromSoAAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("trackSrc", edm::InputTag("pixelTracksAlpaka"));
  desc.add<edm::InputTag>("pixelRecHitLegacySrc", edm::InputTag("siPixelRecHitsPreSplitting"));
  desc.add<int>("minNumberOfHits", 0);
  desc.add<std::string>("minQuality", "loose");

  desc.add<edm::InputTag>("hitModuleStartSrc", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
  desc.add<bool>("useStripHits", false);
  desc.add<edm::InputTag>("stripRecHitLegacySrc", edm::InputTag("siStripMatchedRecHits", "matchedRecHit"));

  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void PixelTrackProducerFromSoAAlpaka<TrackerTraits>::produce(edm::StreamID streamID,
                                                             edm::Event &iEvent,
                                                             const edm::EventSetup &iSetup) const {
  // enum class Quality : uint8_t { bad = 0, edup, dup, loose, strict, tight, highPurity };
  reco::TrackBase::TrackQuality recoQuality[] = {reco::TrackBase::undefQuality,
                                                 reco::TrackBase::undefQuality,
                                                 reco::TrackBase::discarded,
                                                 reco::TrackBase::loose,
                                                 reco::TrackBase::tight,
                                                 reco::TrackBase::tight,
                                                 reco::TrackBase::highPurity};
  assert(reco::TrackBase::highPurity == recoQuality[int(pixelTrack::Quality::highPurity)]);

#ifdef GPU_DEBUG
  std::cout << "Converting soa helix in reco tracks" << std::endl;
#endif

  auto indToEdmP = std::make_unique<IndToEdm>();
  auto &indToEdm = *indToEdmP;

  auto const &idealField = iSetup.getData(idealMagneticFieldToken_);

  pixeltrackfitting::TracksWithRecHits tracks;

  auto const &httopo = iSetup.getData(ttTopoToken_);

  const auto &bsh = iEvent.get(tBeamSpot_);
  GlobalPoint bs(bsh.x0(), bsh.y0(), bsh.z0());

  edm::ESHandle<TrackerGeometry> theTrackerGeometry = iSetup.getHandle(geomToken_);

  auto const &pixelRecHitsDSV = iEvent.get(cpuPixelHits_);
  auto const &pixelRechits = pixelRecHitsDSV.data();
  auto nPixelHits = pixelRechits.size();

  auto const &hitsModuleStart = iEvent.get(hmsToken_);

  size_t nStripHits = 0;
  const edmNew::DetSetVector<SiStripMatchedRecHit2D> *stripRechitsDSV = nullptr;

  if (useStripHits_) {
    stripRechitsDSV = &iEvent.get(cpuStripHits_);
    nStripHits = hitsModuleStart[TrackerTraits::numberOfModules] - hitsModuleStart[TrackerTraits::numberOfPixelModules];
  }

  size_t nhits = nPixelHits + nStripHits;

  std::vector<TrackingRecHit const *> hitmap(nhits, nullptr);
  // std::vector<int> counter(nhits, 0);

  for (const auto &moduleHits : pixelRecHitsDSV) {
    auto *det = theTrackerGeometry->idToDet(moduleHits.detId());
    const auto detI = det->index();
    for (const auto &hit : moduleHits) {
      auto const &clus = hit.firstClusterRef();
      auto const idx = hitsModuleStart[detI] + clus.pixelCluster().originalId();

      if (idx >= hitsModuleStart[detI + 1]) {
        std::cout << "excess pixel hit" << std::endl;
        continue;
      }
      if (idx >= hitmap.size())
        hitmap.resize(idx + 256, nullptr);  /// only in case of hit overflow in one module

      assert(nullptr == hitmap[idx]);

      hitmap[idx] = &hit;
      // ++counter[idx];
    }
  }

  // for (auto const &hit : pixelRechits) {
  //   auto const &thit = static_cast<BaseTrackerRecHit const &>(hit);
  //   auto const detI = thit.det()->index();
  //   auto const &clus = thit.firstClusterRef();
  //   assert(clus.isPixel());
  //   auto const idx = hitsModuleStart[detI] + clus.pixelCluster().originalId();
  //   if (idx >= hitmap.size())
  //     hitmap.resize(idx + 256, nullptr);  // only in case of hit overflow in one module

  //   if (nullptr != hitmap[idx])
  //     throw std::runtime_error("duplicate hit id: " + std::to_string(idx));
  //   hitmap[idx] = &hit;
  //   ++counter[idx];
  // }

  if (useStripHits_) {
    for (const auto &moduleHits : *stripRechitsDSV) {
      const GluedGeomDet *theStripDet =
          dynamic_cast<const GluedGeomDet *>(theTrackerGeometry->idToDet(moduleHits[0].geographicalId()));
      int moduleIdx = TrackerTraits::mapIndex(theStripDet->stereoDet()->index());
      if (moduleIdx >= TrackerTraits::numberOfModules)
        continue;
      for (auto i = 0u; i < moduleHits.size(); ++i) {
        auto j = hitsModuleStart[moduleIdx] + i;
        hitmap[j] = &*(moduleHits.begin() + i);
        // ++counter[j];
      }
    }
  }

#ifdef GPU_DEBUG
  std::cout << "hitmap nulls:" << std::count(hitmap.begin(), hitmap.end(), nullptr) << std::endl;
#endif
  std::vector<const TrackingRecHit *> hits;
  hits.reserve(5);  //TODO: Move to the average depending on tracker traits

  auto const &tsoa = iEvent.get(tokenTrack_);
  auto const *quality = tsoa.view().quality();
  auto const &hitIndices = tsoa.view().hitIndices();
  auto nTracks = tsoa.view().nTracks();

#ifdef GPU_DEBUG
  std::cout << "max hit index = " << *std::max_element(hitIndices.begin(), hitIndices.end()) << std::endl;
  std::cout << "hitmap.size() = " << hitmap.size() << std::endl;
#endif

  tracks.reserve(nTracks);

  int32_t nt = 0;

  //sort index by pt
  std::vector<int32_t> sortIdxs(nTracks);
  std::iota(sortIdxs.begin(), sortIdxs.end(), 0);
  // sort good-quality tracks by pt, keep bad-quality tracks at the bottom
  std::sort(sortIdxs.begin(), sortIdxs.end(), [&](int32_t const i1, int32_t const i2) {
    if (quality[i1] >= minQuality_ && quality[i2] >= minQuality_)
      return tsoa.view()[i1].pt() > tsoa.view()[i2].pt();
    else
      return quality[i1] > quality[i2];
  });

  //store the index of the SoA: indToEdm[index_SoAtrack] -> index_edmTrack (if it exists)
  indToEdm.resize(sortIdxs.size(), -1);
  for (const auto &it : sortIdxs) {
    auto nHits = TracksHelpers::nHits(tsoa.view(), it);

    assert(nHits >= 3);
    auto q = quality[it];
#ifdef GPU_DEBUG
    std::cout << " nHits " << nHits << " quality: " << int(q) << std::endl;
#endif
    if (q < minQuality_)
      continue;
    if (nHits < minNumberOfHits_)  //move to nLayers?
      continue;
    indToEdm[it] = nt;
    ++nt;

    hits.resize(nHits);
    auto b = hitIndices.begin(it);
    for (int iHit = 0; iHit < nHits; ++iHit)
      hits[iHit] = hitmap[*(b + iHit)];

    // mind: this values are respect the beamspot!
    float chi2 = tsoa.view()[it].chi2();
    float phi = reco::phi(tsoa.view(), it);

    riemannFit::Vector5d ipar, opar;
    riemannFit::Matrix5d icov, ocov;
    TracksHelpers::template copyToDense<riemannFit::Vector5d, riemannFit::Matrix5d>(tsoa.view(), ipar, icov, it);
    riemannFit::transformToPerigeePlane(ipar, icov, opar, ocov);

    LocalTrajectoryParameters lpar(opar(0), opar(1), opar(2), opar(3), opar(4), 1.);
    AlgebraicSymMatrix55 m;
    for (int i = 0; i < 5; ++i)
      for (int j = i; j < 5; ++j)
        m(i, j) = ocov(i, j);

    float sp = std::sin(phi);
    float cp = std::cos(phi);
    Surface::RotationType rot(sp, -cp, 0, 0, 0, -1.f, cp, sp, 0);

    Plane impPointPlane(bs, rot);
    GlobalTrajectoryParameters gp(
        impPointPlane.toGlobal(lpar.position()), impPointPlane.toGlobal(lpar.momentum()), lpar.charge(), &idealField);
    JacobianLocalToCurvilinear jl2c(impPointPlane, lpar, idealField);

    AlgebraicSymMatrix55 mo = ROOT::Math::Similarity(jl2c.jacobian(), m);

    int ndof = 2 * hits.size() - 5;
    chi2 = chi2 * ndof;
    GlobalPoint vv = gp.position();
    math::XYZPoint pos(vv.x(), vv.y(), vv.z());
    GlobalVector pp = gp.momentum();
    math::XYZVector mom(pp.x(), pp.y(), pp.z());

    auto track = std::make_unique<reco::Track>(chi2, ndof, pos, mom, gp.charge(), CurvilinearTrajectoryError(mo));
#ifdef GPU_DEBUG
    std::cout << "chi2 " << chi2 << " ndof: " << ndof << "pos " << pos << " mom " << mom << " gp.charge() "
              << gp.charge() << std::endl;
#endif
    // bad and edup not supported as fit not present or not reliable
    auto tkq = recoQuality[int(q)];
    track->setQuality(tkq);
    // loose,tight and HP are inclusive
    if (reco::TrackBase::highPurity == tkq) {
      track->setQuality(reco::TrackBase::tight);
      track->setQuality(reco::TrackBase::loose);
    } else if (reco::TrackBase::tight == tkq) {
      track->setQuality(reco::TrackBase::loose);
    }
    track->setQuality(tkq);
    // filter???
    tracks.emplace_back(track.release(), hits);
  }
#ifdef GPU_DEBUG
  std::cout << "processed " << nt << " good tuples " << tracks.size() << " out of " << indToEdm.size() << std::endl;
#endif

  // store tracks
  storeTracks(iEvent, tracks, httopo);
  iEvent.put(std::move(indToEdmP));
}

using PixelTrackProducerFromSoAAlpakaPhase1 = PixelTrackProducerFromSoAAlpaka<pixelTopology::Phase1>;
using PixelTrackProducerFromSoAAlpakaPhase2 = PixelTrackProducerFromSoAAlpaka<pixelTopology::Phase2>;
using PixelTrackProducerFromSoAAlpakaHIonPhase1 = PixelTrackProducerFromSoAAlpaka<pixelTopology::HIonPhase1>;
using PixelTrackProducerFromSoAAlpakaPhase1Strip = PixelTrackProducerFromSoAAlpaka<pixelTopology::Phase1Strip>;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelTrackProducerFromSoAAlpakaPhase1);
DEFINE_FWK_MODULE(PixelTrackProducerFromSoAAlpakaPhase2);
DEFINE_FWK_MODULE(PixelTrackProducerFromSoAAlpakaHIonPhase1);
DEFINE_FWK_MODULE(PixelTrackProducerFromSoAAlpakaPhase1Strip);
