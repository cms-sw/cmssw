// -*- C++ -*-
//
// Package:    Validation/RecoTrack
// Class:      SimPixelTrackFromRecoTrackProducer
//

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/IndexSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/SimDoublets.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParamsHost.h"
#include "RecoLocalTracker/Records/interface/PixelCPEFastParamsRecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <memory>
#include <typeinfo>

/** Class: SimPixelTrackFromRecoTrackProducer
 * 
 * @brief Produces SimPixelTracks (MC-info based PixelRecHit doublets) from reconstructed tracks.
 *
 * @author Jan Schulz (jan.gerrit.schulz@cern.ch)
 * @date July 2025
 */
template <typename TrackerTraits>
class SimPixelTrackFromRecoTrackProducer : public edm::stream::EDProducer<> {
public:
  explicit SimPixelTrackFromRecoTrackProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

private:
  TrackingParticleSelector trackingParticleSelector;
  pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const* __restrict__ cpeParams_ = nullptr;
  const TrackerTopology* trackerTopology_ = nullptr;
  const TrackerGeometry* trackerGeometry_ = nullptr;

  bool includeFakeTracks_;  // if true, fake RecoTracks (such without any associated TrackingParticle) will be included
  bool includeTrueTracks_;  // if true, true RecoTracks (such with an associated TrackingParticle) will be included

  // tokens for ClusterParameterEstimator, tracker topology, ect.
  const edm::ESGetToken<PixelCPEFastParamsHost<TrackerTraits>, PixelCPEFastParamsRecord> cpe_getToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topology_getToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometry_getToken_;
  const edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHits_getToken_;
  const edm::EDGetTokenT<TrackingParticleCollection> trackingParticles_getToken_;
  const edm::EDGetTokenT<edm::View<reco::Track>> tracks_getToken_;
  const edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> trackAssociator_getToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpot_getToken_;
  const edm::EDPutTokenT<SimDoubletsCollection> simPixelTracks_putToken_;
};

namespace simdoublets {
  // function that determines the layerId from the detId for Phase 2
  template <typename TrackerTraits>
  unsigned int getLayerId(DetId const& detId, const TrackerTopology* trackerTopology) {
    // number of barrel layers
    constexpr unsigned int numBarrelLayers{4};
    // number of disks per endcap
    constexpr unsigned int numEndcapDisks = (TrackerTraits::numberOfLayers - numBarrelLayers) / 2;

    // set default to 999 (invalid)
    unsigned int layerId{99};

    if (detId.subdetId() == PixelSubdetector::PixelBarrel) {
      // subtract 1 in the barrel to get, e.g. for Phase 2, from (1,4) to (0,3)
      layerId = trackerTopology->pxbLayer(detId) - 1;
    } else if (detId.subdetId() == PixelSubdetector::PixelEndcap) {
      if (trackerTopology->pxfSide(detId) == 1) {
        // add offset in the backward endcap to get, e.g. for Phase 2, from (1,12) to (16,27)
        layerId = trackerTopology->pxfDisk(detId) + numBarrelLayers + numEndcapDisks - 1;
      } else {
        // add offest in the forward endcap to get, e.g. for Phase 2, from (1,12) to (4,15)
        layerId = trackerTopology->pxfDisk(detId) + numBarrelLayers - 1;
      }
    } else if (detId.subdetId() == StripSubdetector::TOB) {
      layerId = trackerTopology->getOTLayerNumber(detId) + 27;
    }
    // return the determined Id
    return layerId;
  }

  // function that determines the cluster size of a Pixel RecHit in local y direction
  // according to the formula used in Patatrack reconstruction
  int clusterYSize(OmniClusterRef::ClusterPixelRef const cluster, uint16_t const pixmx, int const maxCol) {
    // check if the cluster lies at the y-edge of the module
    if (cluster->minPixelCol() == 0 || cluster->maxPixelCol() == maxCol) {
      // if so, return -1
      return -1;
    }

    // column span (span of cluster in y direction)
    int span = cluster->colSpan();

    // total charge of the first and last column of digis respectively
    int q_firstCol = 0;
    int q_lastCol = 0;

    // loop over the pixels/digis of the cluster and update the charges of first and last column
    int offset;
    for (int i{0}; i < cluster->size(); i++) {
      offset = cluster->pixelOffset()[2 * i + 1];

      // check if pixel is in first column and eventually update the charge
      if (offset == 0) {
        q_firstCol += std::min(cluster->pixelADC()[i], pixmx);
      }
      // check if pixel is in last column and eventually update the charge
      if (offset == span) {
        q_lastCol += std::min(cluster->pixelADC()[i], pixmx);
      }
    }

    // calculate the unbalance term
    int unbalance = 8. * std::abs(float(q_firstCol - q_lastCol)) / float(q_firstCol + q_lastCol);

    // calculate the cluster size
    int clusterYSize = 8 * (span + 1) - unbalance;
    return clusterYSize;
  }
}  // namespace simdoublets

// constructor
template <typename TrackerTraits>
SimPixelTrackFromRecoTrackProducer<TrackerTraits>::SimPixelTrackFromRecoTrackProducer(const edm::ParameterSet& pSet)
    : includeFakeTracks_(pSet.getParameter<bool>("includeFakeTracks")),
      includeTrueTracks_(pSet.getParameter<bool>("includeTrueTracks")),
      cpe_getToken_(esConsumes(edm::ESInputTag("", pSet.getParameter<std::string>("CPE")))),
      topology_getToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      geometry_getToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      pixelRecHits_getToken_(consumes(pSet.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      trackingParticles_getToken_(consumes(pSet.getParameter<edm::InputTag>("trackingParticleSrc"))),
      tracks_getToken_(consumes<edm::View<reco::Track>>(pSet.getParameter<edm::InputTag>("trackSrc"))),
      trackAssociator_getToken_(consumes<reco::TrackToTrackingParticleAssociator>(
          pSet.getUntrackedParameter<edm::InputTag>("trackAssociator"))),
      beamSpot_getToken_(consumes(pSet.getParameter<edm::InputTag>("beamSpotSrc"))),
      simPixelTracks_putToken_(produces<SimDoubletsCollection>()) {
  // initialize the selector for TrackingParticles used to create SimDoublets
  const edm::ParameterSet& pSetTPSel = pSet.getParameter<edm::ParameterSet>("TrackingParticleSelectionConfig");
  trackingParticleSelector = TrackingParticleSelector(pSetTPSel.getParameter<double>("ptMin"),
                                                      pSetTPSel.getParameter<double>("ptMax"),
                                                      pSetTPSel.getParameter<double>("minRapidity"),
                                                      pSetTPSel.getParameter<double>("maxRapidity"),
                                                      pSetTPSel.getParameter<double>("tip"),
                                                      pSetTPSel.getParameter<double>("lip"),
                                                      pSetTPSel.getParameter<int>("minHit"),
                                                      pSetTPSel.getParameter<bool>("signalOnly"),
                                                      pSetTPSel.getParameter<bool>("intimeOnly"),
                                                      pSetTPSel.getParameter<bool>("chargedOnly"),
                                                      pSetTPSel.getParameter<bool>("stableOnly"),
                                                      pSetTPSel.getParameter<std::vector<int>>("pdgId"),
                                                      pSetTPSel.getParameter<bool>("invertRapidityCut"),
                                                      pSetTPSel.getParameter<double>("minPhi"),
                                                      pSetTPSel.getParameter<double>("maxPhi"));
}

// dummy fillDescription
template <typename TrackerTraits>
void SimPixelTrackFromRecoTrackProducer<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
}

// Phase 2 fillDescription
template <>
void SimPixelTrackFromRecoTrackProducer<pixelTopology::Phase2>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // cluster parameter estimator
  std::string cpe = "PixelCPEFastParams";
  cpe += pixelTopology::Phase2::nameModifier;
  desc.add<std::string>("CPE", cpe)
      ->setComment("Cluster Parameter Estimator (needed for calculating the cluster size)");

  // sources for cluster-TrackingParticle association, TrackingParticles, RecHits and the beamspot
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("hltSiPixelRecHits"));
  desc.add<edm::InputTag>("trackSrc", edm::InputTag("hltPhase2PixelTracks"));
  desc.addUntracked<edm::InputTag>("trackAssociator", edm::InputTag("hltTrackAssociatorByHits"));
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("beamSpotSrc", edm::InputTag("hltOnlineBeamSpot"));

  // Extension settings
  desc.add<int>("numLayersOT", 0)->setComment("Number of additional layers from the OT extension.");
  desc.add<bool>("includeFakeTracks", true)
      ->setComment("If true, fake RecoTracks (such without any associated TrackingParticle) will be included.");
  desc.add<bool>("includeTrueTracks", true)
      ->setComment("If true, true RecoTracks (such with an associated TrackingParticle) will be included.");

  // parameter set for the selection of TrackingParticles that will be used for SimHitDoublets
  edm::ParameterSetDescription descTPSelector;
  descTPSelector.add<double>("ptMin", 1e-3);
  descTPSelector.add<double>("ptMax", 1e100);
  descTPSelector.add<double>("minRapidity", -4.5);
  descTPSelector.add<double>("maxRapidity", 4.5);
  descTPSelector.add<double>("tip", 20.);  // NOTE: differs from HLT MultiTrackValidator
  descTPSelector.add<double>("lip", 300.);
  descTPSelector.add<int>("minHit", 0);
  descTPSelector.add<bool>("signalOnly", false);
  descTPSelector.add<bool>("intimeOnly", false);
  descTPSelector.add<bool>("chargedOnly", true);
  descTPSelector.add<bool>("stableOnly", false);
  descTPSelector.add<std::vector<int>>("pdgId", {});
  descTPSelector.add<bool>("invertRapidityCut", false);
  descTPSelector.add<double>("minPhi", -3.2);
  descTPSelector.add<double>("maxPhi", 3.2);
  desc.add<edm::ParameterSetDescription>("TrackingParticleSelectionConfig", descTPSelector);

  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void SimPixelTrackFromRecoTrackProducer<TrackerTraits>::beginRun(const edm::Run& run,
                                                                 const edm::EventSetup& eventSetup) {}

template <typename TrackerTraits>
void SimPixelTrackFromRecoTrackProducer<TrackerTraits>::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  // -------------------------------------------
  //  Get data from event
  // -------------------------------------------
  // get TrackerTopology and TrackerGeometry
  trackerTopology_ = &eventSetup.getData(topology_getToken_);
  trackerGeometry_ = &eventSetup.getData(geometry_getToken_);

  // get tracks
  edm::Handle<edm::View<reco::Track>> trackCollectionHandle;
  if (!event.getByToken(tracks_getToken_, trackCollectionHandle)) {
    return;
  }
  const edm::View<reco::Track>& trackCollection = *trackCollectionHandle;

  // get track to TrackingParticle association
  auto const& associatorByHits = event.get(trackAssociator_getToken_);
  // get TrackingParticles
  auto TPCollectionH = event.getHandle(trackingParticles_getToken_);

  // get the pixel RecHit collection from the event
  auto const& hits = event.get(pixelRecHits_getToken_); 

  // get cluster parameter estimate
  auto& cpe = eventSetup.getData(cpe_getToken_);
  cpeParams_ = cpe.data();

  // get beamspot from the event
  edm::Handle<reco::BeamSpot> beamSpot;
  event.getByToken(beamSpot_getToken_, beamSpot);
  if (!beamSpot.isValid()) {
    return;
  }

  // create track references
  edm::RefToBaseVector<reco::Track> trackRefs;
  for (edm::View<reco::Track>::size_type i = 0; i < trackCollection.size(); ++i) {
    trackRefs.push_back(trackCollection.refAt(i));
  }
  // select reasonable TrackingParticles
  TrackingParticleRefVector tpCollection;
  for (size_t i = 0, size = TPCollectionH->size(); i < size; ++i) {
    auto tp = TrackingParticleRef(TPCollectionH, i);
    if (trackingParticleSelector(*tp)) {
      tpCollection.push_back(tp);
    }
  }
  // associate Tracks and TrackingParticles
  reco::RecoToSimCollection recSimColl = associatorByHits.associateRecoToSim(trackRefs, tpCollection);
  reco::SimToRecoCollection simRecColl = associatorByHits.associateSimToReco(trackRefs, tpCollection);

  // -------------------------------------------
  //  Build SimPixelTracks
  // -------------------------------------------
  // create collection of SimPixelTracks
  // each element will correspond to one selected track
  SimDoubletsCollection simPixelTrackCollection;

  // initialize a couple of variables used in the following loop
  unsigned int detId, layerId, maxCol;
  uint16_t pixmx;
  int moduleId, clusterYSize {-1};

  // loop over Tracks
  for (auto const& track : trackRefs) {
    // check if track is fake or true
    bool isFake{true};
    auto foundTP = recSimColl.find(track);
    if (foundTP != recSimColl.end()) {
      isFake = (foundTP->val.empty());
    }
    // check if we want to skip this track
    if (isFake && (!includeFakeTracks_))
      continue;
    if (!isFake && (!includeTrueTracks_))
      continue;

    // create SimPixelTrack
    simPixelTrackCollection.emplace_back(SimDoublets(*beamSpot));

    // loop over the RecHits of that Track
    for (auto const& recHit : track->recHits()) {
      detId = recHit->geographicalId().rawId();
      auto globalPos = recHit->globalPosition();
      DetId detIdObject(detId);

      // determine layer Id from detector Id
      layerId = simdoublets::getLayerId<TrackerTraits>(detId, trackerTopology_);

      // determine the module Id
      // const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
      moduleId = trackerGeometry_->idToDetUnit(detIdObject)->index();
      if (layerId < 28) {
        // get CPE parameters for the given module that are used when determining the cluster size:
        // 1. maximum charge per pixel considered for calculating the inbalance term
        pixmx = cpeParams_->detParams(moduleId).pixmx;
        // 2. the index of the boundary column to check if a cluster lies at the module edge
        maxCol = cpeParams_->detParams(moduleId).nCols - 1;
        // 3. find the RecHit in the collection to get the cluster reference
        for (auto const& hit : hits[detId]) {
          auto const hitGlobalPos = hit.globalPosition();
          if ((std::abs(hitGlobalPos.x() - globalPos.x()) < 1e-4) && (std::abs(hitGlobalPos.y() - globalPos.y()) < 1e-4) &&
            (std::abs(hitGlobalPos.z() - globalPos.z()) < 1e-4)) {
          // determine the cluster size of the RecHit
          clusterYSize = simdoublets::clusterYSize(hit.cluster(), pixmx, maxCol);
          break;
        }
        }
      } else {
        clusterYSize = -1;
      }

      // add RecHit to SimPixelTrack
      simPixelTrackCollection.back().addRecHit(*recHit, layerId, clusterYSize, detId, moduleId);
    }  // end loop over RecHits

    // sort the RecHits according to their position
    simPixelTrackCollection.back().sortRecHits(track->vx(), track->vy(), track->vz());
  }  // end loop over Tracks

  // put the produced SimDoublets collection in the event
  event.emplace(simPixelTracks_putToken_, std::move(simPixelTrackCollection));
}

// define two plugins for Phase 2
using SimPixelTrackFromRecoTrackProducerPhase2 = SimPixelTrackFromRecoTrackProducer<pixelTopology::Phase2>;

DEFINE_FWK_MODULE(SimPixelTrackFromRecoTrackProducerPhase2);