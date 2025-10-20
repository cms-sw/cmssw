// -*- C++ -*-
//
// Package:    SimTracker/TrackerHitAssociation
// Class:      SimPixelTrackProducer
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
#include "SimTracker/TrackerHitAssociation/interface/SimPixelTrackTools.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/SimPixelTrack.h"
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

/** Class: SimPixelTrackProducer
 * 
 * @brief Produces SimPixelTracks (MC-info based PixelRecHit doublets) for selected TrackingParticles.
 *
 * SimDoublets represent the true doublets of RecHits that a simulated particle (TrackingParticle) 
 * created in the pixel detector. They can be used to analyze cuts which are applied in the reconstruction
 * when producing doublets as the first part of patatrack pixel tracking.
 *
 * The SimPixelTrack are produced in the following way:
 * 1. We select reasonable TrackingParticles according to the criteria given in the config file as 
 *    "TrackingParticleSelectionConfig".
 * 2. For each selected particle, we create and append a new SimPixelTrack object to the SimPixelTrackCollection.
 * 3. We loop over all RecHits in the pixel tracker and check if the given RecHit is associated to one of
 *    the selected particles (association via TP to cluster association). If it is, we add a RecHit reference
 *    to the respective SimDoublet.
 * 4. In the end, we sort the RecHits in each SimPixelTrack object according to their global position.
 *
 * @author Jan Schulz (jan.gerrit.schulz@cern.ch)
 * @date January 2025
 */
template <typename TrackerTraits>
class SimPixelTrackProducer : public edm::stream::EDProducer<> {
public:
  explicit SimPixelTrackProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

private:
  TrackingParticleSelector trackingParticleSelector;
  pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const* __restrict__ cpeParams_ = nullptr;
  const TrackerTopology* trackerTopology_ = nullptr;
  const TrackerGeometry* trackerGeometry_ = nullptr;

  bool includeOTBarrel_;       // if true, OT barrel layers are considered for CA extension
  bool includeOTDisks_;        // if true, OT disks layers are considered for CA extension
  bool dropEvenLayerRecHits_;  // if true, no RecHits from even layers are considered
  bool dropOddLayerRecHits_;   // if true, no RecHits from odd layers are considered

  // tokens for ClusterParameterEstimator, tracker topology, ect.
  const edm::ESGetToken<PixelCPEFastParamsHost<TrackerTraits>, PixelCPEFastParamsRecord> cpe_getToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topology_getToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometry_getToken_;
  const edm::EDGetTokenT<ClusterTPAssociation> clusterTPAssociation_getToken_;
  const edm::EDGetTokenT<TrackingParticleCollection> trackingParticles_getToken_;
  const edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHits_getToken_;
  const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> otRecHits_getToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpot_getToken_;
  const edm::EDPutTokenT<SimPixelTrackCollection> simPixelTracks_putToken_;
};

// constructor
template <typename TrackerTraits>
SimPixelTrackProducer<TrackerTraits>::SimPixelTrackProducer(const edm::ParameterSet& pSet)
    : includeOTBarrel_(pSet.getParameter<bool>("includeOTBarrel")),
      includeOTDisks_(pSet.getParameter<bool>("includeOTDisks")),
      dropEvenLayerRecHits_(pSet.getParameter<bool>("dropEvenLayerRecHits")),
      dropOddLayerRecHits_(pSet.getParameter<bool>("dropOddLayerRecHits")),
      cpe_getToken_(esConsumes(edm::ESInputTag("", pSet.getParameter<std::string>("CPE")))),
      topology_getToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      geometry_getToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      clusterTPAssociation_getToken_(
          consumes<ClusterTPAssociation>(pSet.getParameter<edm::InputTag>("clusterTPAssociationSrc"))),
      trackingParticles_getToken_(consumes(pSet.getParameter<edm::InputTag>("trackingParticleSrc"))),
      pixelRecHits_getToken_(consumes(pSet.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      otRecHits_getToken_(consumes(pSet.getParameter<edm::InputTag>("outerTrackerRecHitSrc"))),
      beamSpot_getToken_(consumes(pSet.getParameter<edm::InputTag>("beamSpotSrc"))),
      simPixelTracks_putToken_(produces<SimPixelTrackCollection>()) {
  // initialize the selector for TrackingParticles used to create SimPixelTracks
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
void SimPixelTrackProducer<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {}

// Phase 1 fillDescription
template <>
void SimPixelTrackProducer<pixelTopology::Phase1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // cluster parameter estimator
  std::string cpe = "PixelCPEFastParams";
  cpe += pixelTopology::Phase1::nameModifier;
  desc.add<std::string>("CPE", cpe)
      ->setComment("Cluster Parameter Estimator (needed for calculating the cluster size)");

  // sources for cluster-TrackingParticle association, TrackingParticles, RecHits and the beamspot
  desc.add<edm::InputTag>("clusterTPAssociationSrc", edm::InputTag("hltTPClusterProducer"));
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("hltSiPixelRecHits"));
  desc.add<edm::InputTag>("outerTrackerRecHitSrc", edm::InputTag("hltSiPhase2RecHits"));
  desc.add<edm::InputTag>("beamSpotSrc", edm::InputTag("hltOnlineBeamSpot"));

  // Extension settings
  desc.add<bool>("includeOTBarrel", false)->setComment("If true, add barrel layers from the OT extension.");
  desc.add<bool>("includeOTDisks", false)->setComment("If true, add disk layers from the OT extension.");
  desc.add<bool>("dropEvenLayerRecHits", false)
      ->setComment("If true, the RecHits in layers with even index are dropped when building the SimNtuplets.");
  desc.add<bool>("dropOddLayerRecHits", false)
      ->setComment("If true, the RecHits in layers with odd index are dropped when building the SimNtuplets.");

  // parameter set for the selection of TrackingParticles that will be used for SimHitDoublets
  edm::ParameterSetDescription descTPSelector;
  descTPSelector.add<double>("ptMin", 0.9);
  descTPSelector.add<double>("ptMax", 1e100);
  descTPSelector.add<double>("minRapidity", -3.);
  descTPSelector.add<double>("maxRapidity", 3.);
  descTPSelector.add<double>("tip", 2.5);
  descTPSelector.add<double>("lip", 30.);
  descTPSelector.add<int>("minHit", 0);
  descTPSelector.add<bool>("signalOnly", true);
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

// Phase 2 fillDescription
template <>
void SimPixelTrackProducer<pixelTopology::Phase2>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // cluster parameter estimator
  std::string cpe = "PixelCPEFastParams";
  cpe += pixelTopology::Phase2::nameModifier;
  desc.add<std::string>("CPE", cpe)
      ->setComment("Cluster Parameter Estimator (needed for calculating the cluster size)");

  // sources for cluster-TrackingParticle association, TrackingParticles, RecHits and the beamspot
  desc.add<edm::InputTag>("clusterTPAssociationSrc", edm::InputTag("hltTPClusterProducer"));
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("hltSiPixelRecHits"));
  desc.add<edm::InputTag>("outerTrackerRecHitSrc", edm::InputTag("hltSiPhase2RecHits"));
  desc.add<edm::InputTag>("beamSpotSrc", edm::InputTag("hltOnlineBeamSpot"));

  // Extension settings
  desc.add<bool>("includeOTBarrel", false)->setComment("If true, add barrel layers from the OT extension.");
  desc.add<bool>("includeOTDisks", false)->setComment("If true, add disk layers from the OT extension.");
  desc.add<bool>("dropEvenLayerRecHits", false)
      ->setComment("If true, the RecHits in layers with even index are dropped when building the SimNtuplets.");
  desc.add<bool>("dropOddLayerRecHits", false)
      ->setComment("If true, the RecHits in layers with odd index are dropped when building the SimNtuplets.");

  // parameter set for the selection of TrackingParticles that will be used for SimHitDoublets
  edm::ParameterSetDescription descTPSelector;
  descTPSelector.add<double>("ptMin", 0.9);
  descTPSelector.add<double>("ptMax", 1e100);
  descTPSelector.add<double>("minRapidity", -4.5);
  descTPSelector.add<double>("maxRapidity", 4.5);
  descTPSelector.add<double>("tip", 2.5);
  descTPSelector.add<double>("lip", 30.);
  descTPSelector.add<int>("minHit", 0);
  descTPSelector.add<bool>("signalOnly", true);
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
void SimPixelTrackProducer<TrackerTraits>::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) {}

template <typename TrackerTraits>
void SimPixelTrackProducer<TrackerTraits>::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  // get TrackerTopology and TrackerGeometry
  trackerTopology_ = &eventSetup.getData(topology_getToken_);
  trackerGeometry_ = &eventSetup.getData(geometry_getToken_);

  // get cluster parameter estimate
  auto& cpe = eventSetup.getData(cpe_getToken_);
  cpeParams_ = cpe.data();

  // get cluster to TrackingParticle association
  ClusterTPAssociation const& clusterTPAssociation = event.get(clusterTPAssociation_getToken_);

  // get the pixel RecHit collection from the event
  edm::Handle<SiPixelRecHitCollection> hits;
  event.getByToken(pixelRecHits_getToken_, hits);
  if (!hits.isValid()) {
    return;
  }

  // get the Outer Tracker RecHit collection from the event
  edm::Handle<Phase2TrackerRecHit1DCollectionNew> hitsOT;
  event.getByToken(otRecHits_getToken_, hitsOT);
  if (((includeOTBarrel_) || (includeOTDisks_)) && (!hitsOT.isValid())) {
    return;
  }

  // get TrackingParticles from the event
  edm::Handle<TrackingParticleCollection> trackingParticles;
  event.getByToken(trackingParticles_getToken_, trackingParticles);
  if (!trackingParticles.isValid()) {
    return;
  }

  // get beamspot from the event
  edm::Handle<reco::BeamSpot> beamSpot;
  event.getByToken(beamSpot_getToken_, beamSpot);
  if (!beamSpot.isValid()) {
    return;
  }

  // create collection of SimPixelTrack
  // each element will correspond to one selected TrackingParticle
  SimPixelTrackCollection simPixelTrackCollection;

  // loop over TrackingParticles
  for (size_t i = 0; i < trackingParticles->size(); ++i) {
    TrackingParticle const& trackingParticle = trackingParticles->at(i);

    // select reasonable TrackingParticles for the study (e.g., only signal)
    if (trackingParticleSelector(trackingParticle)) {
      simPixelTrackCollection.push_back(SimPixelTrack(TrackingParticleRef(trackingParticles, i), *beamSpot));
    }
  }

  // create a set of the keys of the selected TrackingParticles
  edm::IndexSet selectedTrackingParticleKeys;
  selectedTrackingParticleKeys.reserve(simPixelTrackCollection.size());
  for (const auto& simPixelTrack : simPixelTrackCollection) {
    TrackingParticleRef trackingParticleRef = simPixelTrack.trackingParticle();
    selectedTrackingParticleKeys.insert(trackingParticleRef.key());
  }

  // initialize a couple of counters
  int count_associatedRecHits{0}, count_RecHitsInSimPixelTrack{0};

  // initialize a couple of variables used in the following loop
  unsigned int detId, layerId, maxCol;
  uint16_t pixmx;
  int moduleId, clusterYSize;

  // loop over pixel RecHit collections of the different pixel modules
  for (const auto& detSet : *hits) {
    // get detector Id
    detId = detSet.detId();
    DetId detIdObject(detId);

    // determine layer Id from detector Id
    layerId = simpixeltracks::getLayerId<TrackerTraits>(detId, trackerTopology_);

    // check if we would like to skip
    if (dropEvenLayerRecHits_ && (layerId % 2 == 0)) {
      continue;
    }
    if (dropOddLayerRecHits_ && (layerId % 2 == 1)) {
      continue;
    }

    // determine the module Id
    // const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    moduleId = trackerGeometry_->idToDetUnit(detIdObject)->index();
    // get CPE parameters for the given module that are used when determining the cluster size:
    // 1. maximum charge per pixel considered for calculating the inbalance term
    pixmx = cpeParams_->detParams(moduleId).pixmx;
    // 2. the index of the boundary column to check if a cluster lies at the module edge
    maxCol = cpeParams_->detParams(moduleId).nCols - 1;

    // loop over RecHits
    for (auto const& hit : detSet) {
      // find associated TrackingParticles
      auto range = clusterTPAssociation.equal_range(OmniClusterRef(hit.cluster()));

      // if the RecHit has associated TrackingParticles
      if (range.first != range.second) {
        for (auto assocTrackingParticleIter = range.first; assocTrackingParticleIter != range.second;
             assocTrackingParticleIter++) {
          const TrackingParticleRef assocTrackingParticle = (assocTrackingParticleIter->second);

          // if the associated TrackingParticle is among the selected ones
          if (selectedTrackingParticleKeys.has(assocTrackingParticle.key())) {
            // determine the cluster size of the RecHit
            clusterYSize = simpixeltracks::clusterYSize(hit.cluster(), pixmx, maxCol);
            count_associatedRecHits++;
            // loop over collection of SimPixelTrack and find the one of the associated TrackingParticle
            for (auto& simPixelTrack : simPixelTrackCollection) {
              TrackingParticleRef trackingParticleRef = simPixelTrack.trackingParticle();
              if (assocTrackingParticle.key() == trackingParticleRef.key()) {
                simPixelTrack.addRecHit(hit, layerId, clusterYSize, detId, moduleId);
                count_RecHitsInSimPixelTrack++;
              }
            }
          }
        }
      }
    }  // end loop over RecHits
  }  // end loop over pixel RecHit collections of the different pixel modules

  // default the cluster size for the OT hits to -1 since they will not be
  // considered at any point of the reconstruction anyway at moment
  clusterYSize = -1;

  // loop over Outer Tracker RecHit collections of the different modules
  // if OT layers should be considered
  if ((includeOTBarrel_) || (includeOTDisks_)) {
    for (const auto& detSetOT : *hitsOT) {
      // get detector Id
      detId = detSetOT.detId();
      DetId detIdObject(detId);

      // only use the RecHits if the module is in the accepted range of layers and one of the Phase 2 PS, p-sensor
      if (!(trackerGeometry_->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP))
        continue;

      if (!(detIdObject.subdetId() == SiStripSubdetector::TOB) && !(detIdObject.subdetId() == SiStripSubdetector::TID))
        continue;

      // determine layer Id from detector Id
      layerId = simpixeltracks::getLayerId<TrackerTraits>(detId, trackerTopology_);

      // check if we would like to skip
      if (dropEvenLayerRecHits_ && (layerId % 2 == 0)) {
        continue;
      }
      if (dropOddLayerRecHits_ && (layerId % 2 == 1)) {
        continue;
      }

      // determine the module Id
      moduleId = trackerGeometry_->idToDetUnit(detIdObject)->index();

      // loop over RecHits
      for (auto const& hitOT : detSetOT) {
        // std::cout << "OT RecHit in layer " << layerId << ": " << hitOT.globalPosition() << std::endl;

        // find associated TrackingParticles
        auto range = clusterTPAssociation.equal_range(OmniClusterRef(hitOT.cluster()));

        // if the RecHit has associated TrackingParticles
        if (range.first != range.second) {
          for (auto assocTrackingParticleIter = range.first; assocTrackingParticleIter != range.second;
               assocTrackingParticleIter++) {
            const TrackingParticleRef assocTrackingParticle = (assocTrackingParticleIter->second);

            // if the associated TrackingParticle is among the selected ones
            if (selectedTrackingParticleKeys.has(assocTrackingParticle.key())) {
              // loop over collection of SimPixelTrack and find the one of the associated TrackingParticle
              for (auto& simPixelTrack : simPixelTrackCollection) {
                TrackingParticleRef trackingParticleRef = simPixelTrack.trackingParticle();
                if (assocTrackingParticle.key() == trackingParticleRef.key()) {
                  // add the RecHit to the SimDoublet
                  simPixelTrack.addRecHit(hitOT, layerId, clusterYSize, detId, moduleId);
                }
              }
            }
          }
        }
      }  // end loop over RecHits
    }  // end loop over OT RecHit collections of the different modules
  }

  // loop over collection of SimPixelTrack and sort the RecHits according to their position
  for (auto& simPixelTrack : simPixelTrackCollection) {
    simPixelTrack.sortRecHits();
  }

  LogDebug("SimPixelTrackProducer") << "Size of SiPixelRecHitCollection : " << hits->size() << std::endl;
  LogDebug("SimPixelTrackProducer") << count_associatedRecHits << " of " << hits->size()
                                    << " RecHits are associated to selected TrackingParticles ("
                                    << count_RecHitsInSimPixelTrack - count_associatedRecHits
                                    << " of them were associated multiple times)." << std::endl;
  LogDebug("SimPixelTrackProducer") << "Number of selected TrackingParticles : " << simPixelTrackCollection.size()
                                    << std::endl;
  LogDebug("SimPixelTrackProducer") << "Size of TrackingParticle Collection  : " << trackingParticles->size()
                                    << std::endl;

  // put the produced SimPixelTrack collection in the event
  event.emplace(simPixelTracks_putToken_, std::move(simPixelTrackCollection));
}

// define two plugins for Phase 1 and 2
using SimPixelTrackProducerPhase1 = SimPixelTrackProducer<pixelTopology::Phase1>;
using SimPixelTrackProducerPhase2 = SimPixelTrackProducer<pixelTopology::Phase2>;

DEFINE_FWK_MODULE(SimPixelTrackProducerPhase1);
DEFINE_FWK_MODULE(SimPixelTrackProducerPhase2);
