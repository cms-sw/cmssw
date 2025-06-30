#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
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
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
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
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/FitUtils.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "storeTracks.h"

/**
 * This class creates "legacy" reco::Track
 * objects from the output of SoA CA.
 */

#define GPU_DEBUG

class PixelTrackProducerFromSoAAlpaka : public edm::global::EDProducer<> {
  using TrackSoAHost = reco::TracksHost;
  using HMSstorage = std::vector<uint32_t>;
  using IndToEdm = std::vector<uint32_t>;
  using TrackHitSoA = reco::TrackHitSoA;

public:
  explicit PixelTrackProducerFromSoAAlpaka(const edm::ParameterSet &iConfig);
  ~PixelTrackProducerFromSoAAlpaka() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const override;

  // Event Data tokens
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const edm::EDGetTokenT<TrackSoAHost> trackSoAToken_;
  const edm::EDGetTokenT<SiPixelRecHitCollectionNew> pixelRecHitsToken_;
  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> otRecHitsToken_;
  const edm::EDGetTokenT<HMSstorage> hmsToken_;
  // Event Setup tokens
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> idealMagneticFieldToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;

  int32_t const minNumberOfHits_;
  pixelTrack::Quality const minQuality_;
  const bool useOTExtension_;
};

PixelTrackProducerFromSoAAlpaka::PixelTrackProducerFromSoAAlpaka(const edm::ParameterSet &iConfig)
    : beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      trackSoAToken_(consumes(iConfig.getParameter<edm::InputTag>("trackSrc"))),
      pixelRecHitsToken_(
          consumes<SiPixelRecHitCollectionNew>(iConfig.getParameter<edm::InputTag>("pixelRecHitLegacySrc"))),
      hmsToken_(consumes<HMSstorage>(iConfig.getParameter<edm::InputTag>("pixelRecHitLegacySrc"))),
      idealMagneticFieldToken_(esConsumes()),
      trackerTopologyToken_(esConsumes()),
      trackerGeometryToken_(esConsumes()),
      minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits")),
      minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
      useOTExtension_(iConfig.getParameter<bool>("useOTExtension")) {
  if (minQuality_ == pixelTrack::Quality::notQuality) {
    throw cms::Exception("PixelTrackConfiguration")
        << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
  }
  if (minQuality_ < pixelTrack::Quality::dup) {
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

  // if useOTExtension consume the OT RecHits
  if (useOTExtension_) {
    otRecHitsToken_ =
        consumes<Phase2TrackerRecHit1DCollectionNew>(iConfig.getParameter<edm::InputTag>("outerTrackerRecHitSrc"));
  }
}

void PixelTrackProducerFromSoAAlpaka::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("trackSrc", edm::InputTag("pixelTracksAlpaka"));
  desc.add<edm::InputTag>("pixelRecHitLegacySrc", edm::InputTag("siPixelRecHitsPreSplittingLegacy"));
  desc.add<edm::InputTag>("outerTrackerRecHitSrc", edm::InputTag("hltSiPhase2RecHits"));
  desc.add<int>("minNumberOfHits", 0);
  desc.add<std::string>("minQuality", "loose");
  desc.add<bool>("useOTExtension", false);
  descriptions.addWithDefaultLabel(desc);
}

void PixelTrackProducerFromSoAAlpaka::produce(edm::StreamID streamID,
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

  // index map: trackId(in SoA) -> trackId(in legacy edm)
  auto indToEdmP = std::make_unique<IndToEdm>();
  auto &indToEdm = *indToEdmP;

  auto const &idealField = iSetup.getData(idealMagneticFieldToken_);

  // prepare container for legacy tracks
  pixeltrackfitting::TracksWithRecHits tracks;

  // get trackerTopology and trackerGeometry
  auto const &trackerTopology = iSetup.getData(trackerTopologyToken_);
  const auto &trackerGeometry = &iSetup.getData(trackerGeometryToken_);

  // get beamspot
  const auto &bsh = iEvent.get(beamSpotToken_);
  GlobalPoint bs(bsh.x0(), bsh.y0(), bsh.z0());

  // get the module's starting indices in the hit collection
  auto const &hitsModuleStart = iEvent.get(hmsToken_);

  // get Pixel RecHits
  auto const &pixelRecHitsDSV = iEvent.get(pixelRecHitsToken_);
  auto const &pixelRecHits = pixelRecHitsDSV.data();
  auto const nPixelHits = pixelRecHits.size();

  // get OT RecHits if needed
  size_t nOTHits = 0;
  const Phase2TrackerRecHit1DCollectionNew *otRecHitsDSV = nullptr;
  if (useOTExtension_) {
    otRecHitsDSV = &iEvent.get(otRecHitsToken_);
    nOTHits = otRecHitsDSV->dataSize();
  }

  size_t nHits = nPixelHits + nOTHits;

  // hitmap to go from a unique RecHit identifier to the RecHit in the legacy collection
  // (unique hit identifier is equivalent to the position of the hit in the RecHit SoA)
  std::vector<TrackingRecHit const *> hitmap;
  hitmap.resize(nHits, nullptr);

  // loop over pixel RecHits to fill the hitmap
  for (auto const &pixelHit : pixelRecHits) {
    auto const &thit = static_cast<BaseTrackerRecHit const &>(pixelHit);
    auto const detI = thit.det()->index();
    auto const &clus = thit.firstClusterRef();
    assert(clus.isPixel());

    // get hit identifier as (hit offset of the module) + (hit index in this module)
    auto const idx = hitsModuleStart[detI] + clus.pixelCluster().originalId();

    if (idx >= hitmap.size())
      hitmap.resize(idx + 256, nullptr);  // only in case of hit overflow in one module

    assert(nullptr == hitmap[idx]);
    hitmap[idx] = &pixelHit;
  }

  // function to select only P-hits from the OT barrel
  auto isPinPSinOTBarrel = [&](DetId detId) {
    return (trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP &&
            detId.subdetId() == StripSubdetector::TOB);
  };

  // if OT RecHits are used in PixelTracks, fill the hitmap also with those
  if (useOTExtension_) {
    // The RecHits in the SoA are ordered according to the detUnit->index()
    // of the respective OT module. For this reason, in order to infer the
    // hit position in the SoA, we need to know which  modules are there and
    // how many hits each module has.
    // We use:
    //   - detIdToIndex: map the detId to detUnit->index()
    //     (NOTE: technically, we wouldn't need the detUnit at all if the order
    //     of detIds and detUnit->index is identical, to be checked)
    //   - p_modulesInPSInOTBarrel: set of detUnit->index() of considered modules
    //     (used for finding the "moduleIdOT", position of the module in the SoA)
    //   - moduleStartOT: map from moduleIdOT to position of first RecHit of that
    //     layer in the RecHit SoA

    auto const &detUnits = trackerGeometry->detUnits();
    std::map<uint32_t, uint16_t> detIdToIndex;
    std::set<int> p_modulesInPSInOTBarrel;

    // fill detIdToIndex map and p_modulesInPSInOTBarrel set of modules
    for (auto &detUnit : detUnits) {
      DetId detId(detUnit->geographicalId());
      if (isPinPSinOTBarrel(detId)) {
        detIdToIndex[detUnit->geographicalId()] = detUnit->index();
        p_modulesInPSInOTBarrel.insert(detUnit->index());
      }
    }

    // function to get the "moduleId" of the OT modules
    // (NOT a general CMSSW Id but just the index of the OT module in moduleStartOT)
    auto getModuleIdOT = [&](DetId detId) {
      int index = detIdToIndex[detId];
      auto it = p_modulesInPSInOTBarrel.find(index);
      if (it != p_modulesInPSInOTBarrel.end()) {
        return std::distance(p_modulesInPSInOTBarrel.begin(), it);
      } else {
        return -1L;
      }
    };

    // count hits in all considered OT modules and fill them in moduleStartOT
    // at the position of the subsequent module
    std::vector<int> moduleStartOT;
    moduleStartOT.resize(p_modulesInPSInOTBarrel.size() + 1, 0);
    for (auto const &detSet : *otRecHitsDSV) {
      auto detId = detSet.detId();
      if (isPinPSinOTBarrel(DetId(detId))) {
        int moduleId = getModuleIdOT(detId);
        if (moduleId != -1) {
          moduleStartOT[moduleId + 1] = detSet.size();
        } else {
          assert(0);
        }
      }
    }

    // accumulate the number of hits starting from the number of pixel hits
    // to finalize the actual positions of the layers in the RecHit SoA
    moduleStartOT[0] = nPixelHits;
    std::partial_sum(moduleStartOT.cbegin(), moduleStartOT.cend(), moduleStartOT.begin());

    // perform the exact same loop of how the SoA is initially filled with OT hits
    // and get the index by counting the hits (starting from nPixelHits)
    for (auto const &detSet : *otRecHitsDSV) {
      auto detId = detSet.detId();
      if (isPinPSinOTBarrel(DetId(detId))) {
        for (int idx = moduleStartOT[getModuleIdOT(detId)]; auto const &recHit : detSet) {
          hitmap[idx] = &recHit;
          idx++;
        }
      }
    }
  }

  std::vector<const TrackingRecHit *> hits;
  hits.reserve(5);  //TODO move to a configurable parameter?

  auto const &tsoa = iEvent.get(trackSoAToken_);
  auto const *quality = tsoa.view().quality();
  auto const *hitOffs = tsoa.view().hitOffsets();
  auto const *hitIdxs = tsoa.template view<TrackHitSoA>().id();
  // auto const &hitIndices = tsoa.view().hitIndices();
  auto nTracks = tsoa.view().nTracks();

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

  indToEdm.resize(nTracks, -1);

  // loop over (sorted) tracks
  for (const auto &it : sortIdxs) {
    auto nHits = reco::nHits(tsoa.view(), it);
    assert(nHits >= 3);
    auto q = quality[it];

    // apply cuts on quality and number of hits
    if (q < minQuality_)
      // since the tracks are sorted according to quality,
      // we can break after the first track with low quality
      break;
    if (nHits < minNumberOfHits_)  //move to nLayers?
      continue;

    //store the index of the SoA:
    // indToEdm[index_SoAtrack] -> index_edmTrack (if it exists)
    indToEdm[it] = nt;
    ++nt;

    hits.resize(nHits);
    auto start = (it == 0) ? 0 : hitOffs[it - 1];
    auto end = hitOffs[it];

    for (auto iHit = start; iHit < end; ++iHit)
      hits[iHit - start] = hitmap[hitIdxs[iHit]];

#ifdef CA_DEBUG
    std::cout << "track soa : " << it << " with hits: ";
    for (auto iHit = start; iHit < end; ++iHit)
      std::cout << hitIdxs[iHit] << " - ";
    std::cout << std::endl;
#endif

    // mind: this values are respect the beamspot!
    float chi2 = tsoa.view()[it].chi2();
    float phi = reco::phi(tsoa.view(), it);

    riemannFit::Vector5d ipar, opar;
    riemannFit::Matrix5d icov, ocov;
    reco::copyToDense<riemannFit::Vector5d, riemannFit::Matrix5d>(tsoa.view(), ipar, icov, it);
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
  storeTracks(iEvent, tracks, trackerTopology);
  iEvent.put(std::move(indToEdmP));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelTrackProducerFromSoAAlpaka);

// (also) for HLT migration, could be removed once done
using PixelTrackProducerFromSoAAlpakaPhase1 = PixelTrackProducerFromSoAAlpaka;
using PixelTrackProducerFromSoAAlpakaPhase2 = PixelTrackProducerFromSoAAlpaka;
using PixelTrackProducerFromSoAAlpakaHIonPhase1 = PixelTrackProducerFromSoAAlpaka;

DEFINE_FWK_MODULE(PixelTrackProducerFromSoAAlpakaPhase1);
DEFINE_FWK_MODULE(PixelTrackProducerFromSoAAlpakaPhase2);
DEFINE_FWK_MODULE(PixelTrackProducerFromSoAAlpakaHIonPhase1);
