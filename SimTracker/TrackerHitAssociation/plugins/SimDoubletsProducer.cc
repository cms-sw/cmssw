// -*- C++ -*-
//
// Package:    SimTracker/TrackerHitAssociation
// Class:      SimDoubletsProducer
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

#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/SimDoublets.h"

#include <cstddef>
#include <utility>
#include <vector>
#include <memory>
#include <typeinfo>

/** Class: SimDoubletsProducer
 * 
 * @brief Produces SimDoublets (MC-info based PixelRecHit doublets) for selected TrackingParticles.
 *
 * SimDoublets represent the true doublets of RecHits that a simulated particle (TrackingParticle) 
 * created in the pixel detector. They can be used to analyze cuts which are applied in the reconstruction
 * when producing doublets as the first part of patatrack pixel tracking.
 *
 * The SimDoublets are produced in the following way:
 * 1. We select reasonable TrackingParticles according to the criteria given in the config file as 
 *    "TrackingParticleSelectionConfig".
 * 2. For each selected particle, we create and append a new SimDoublets object to the SimDoubletsCollection.
 * 3. We loop over all RecHits in the pixel tracker and check if the given RecHit is associated to one of
 *    the selected particles (association via TP to cluster association). If it is, we add a RecHit reference
 *    to the respective SimDoublet.
 * 4. In the end, we sort the RecHits in each SimDoublets object according to their global position.
 *
 * @author Jan Schulz (jan.gerrit.schulz@cern.ch)
 * @date January 2025
 */
template <typename TrackerTraits>
class SimDoubletsProducer : public edm::stream::EDProducer<> {
public:
  explicit SimDoubletsProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

private:
  TrackingParticleSelector trackingParticleSelector;
  const TrackerTopology* trackerTopology_ = nullptr;

  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topology_getToken_;
  const edm::EDGetTokenT<ClusterTPAssociation> clusterTPAssociation_getToken_;
  const edm::EDGetTokenT<TrackingParticleCollection> trackingParticles_getToken_;
  const edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHits_getToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpot_getToken_;
  const edm::EDPutTokenT<SimDoubletsCollection> simDoublets_putToken_;
};

namespace simdoublets {
  // function that determines the layerId from the detId for Phase 1 and 2
  template <typename TrackerTraits>
  unsigned int getLayerId(DetId const& detId, const TrackerTopology* trackerTopology) {
    // number of barrel layers
    constexpr unsigned int numBarrelLayers{4};
    // number of disks per endcap
    constexpr unsigned int numEndcapDisks = (TrackerTraits::numberOfLayers - numBarrelLayers) / 2;

    // set default to 999 (invalid)
    unsigned int layerId{999};

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
    }
    // return the determined Id
    return layerId;
  }
}  // namespace simdoublets

// constructor
template <typename TrackerTraits>
SimDoubletsProducer<TrackerTraits>::SimDoubletsProducer(const edm::ParameterSet& pSet)
    : topology_getToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      clusterTPAssociation_getToken_(
          consumes<ClusterTPAssociation>(pSet.getParameter<edm::InputTag>("clusterTPAssociationSrc"))),
      trackingParticles_getToken_(consumes(pSet.getParameter<edm::InputTag>("trackingParticleSrc"))),
      pixelRecHits_getToken_(consumes(pSet.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      beamSpot_getToken_(consumes(pSet.getParameter<edm::InputTag>("beamSpotSrc"))),
      simDoublets_putToken_(produces<SimDoubletsCollection>()) {
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
void SimDoubletsProducer<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {}

// Phase 1 fillDescription
template <>
void SimDoubletsProducer<pixelTopology::Phase1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // sources for cluster-TrackingParticle association, TrackingParticles, RecHits and the beamspot
  desc.add<edm::InputTag>("clusterTPAssociationSrc", edm::InputTag("hltTPClusterProducer"));
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("hltSiPixelRecHits"));
  desc.add<edm::InputTag>("beamSpotSrc", edm::InputTag("hltOnlineBeamSpot"));

  // parameter set for the selection of TrackingParticles that will be used for SimHitDoublets
  edm::ParameterSetDescription descTPSelector;
  descTPSelector.add<double>("ptMin", 0.9);
  descTPSelector.add<double>("ptMax", 1e100);
  descTPSelector.add<double>("minRapidity", -3.);
  descTPSelector.add<double>("maxRapidity", 3.);
  descTPSelector.add<double>("tip", 60.);
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
void SimDoubletsProducer<pixelTopology::Phase2>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // sources for cluster-TrackingParticle association, TrackingParticles, RecHits and the beamspot
  desc.add<edm::InputTag>("clusterTPAssociationSrc", edm::InputTag("hltTPClusterProducer"));
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("hltSiPixelRecHits"));
  desc.add<edm::InputTag>("beamSpotSrc", edm::InputTag("hltOnlineBeamSpot"));

  // parameter set for the selection of TrackingParticles that will be used for SimHitDoublets
  edm::ParameterSetDescription descTPSelector;
  descTPSelector.add<double>("ptMin", 0.9);
  descTPSelector.add<double>("ptMax", 1e100);
  descTPSelector.add<double>("minRapidity", -4.5);
  descTPSelector.add<double>("maxRapidity", 4.5);
  descTPSelector.add<double>("tip", 2.);  // NOTE: differs from HLT MultiTrackValidator
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
void SimDoubletsProducer<TrackerTraits>::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) {}

template <typename TrackerTraits>
void SimDoubletsProducer<TrackerTraits>::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  // get TrackerTopology
  trackerTopology_ = &eventSetup.getData(topology_getToken_);

  // get cluster to TrackingParticle association
  ClusterTPAssociation const& clusterTPAssociation = event.get(clusterTPAssociation_getToken_);

  // get the pixel RecHit collection from the event
  edm::Handle<SiPixelRecHitCollection> hits;
  event.getByToken(pixelRecHits_getToken_, hits);
  if (!hits.isValid()) {
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

  // create collection of SimDoublets
  // each element will correspond to one selected TrackingParticle
  SimDoubletsCollection simDoubletsCollection;

  // loop over TrackingParticles
  for (size_t i = 0; i < trackingParticles->size(); ++i) {
    TrackingParticle const& trackingParticle = trackingParticles->at(i);

    // select reasonable TrackingParticles for the study (e.g., only signal)
    if (trackingParticleSelector(trackingParticle)) {
      simDoubletsCollection.push_back(SimDoublets(TrackingParticleRef(trackingParticles, i), *beamSpot));
    }
  }

  // create a set of the keys of the selected TrackingParticles
  edm::IndexSet selectedTrackingParticleKeys;
  selectedTrackingParticleKeys.reserve(simDoubletsCollection.size());
  for (const auto& simDoublets : simDoubletsCollection) {
    TrackingParticleRef trackingParticleRef = simDoublets.trackingParticle();
    selectedTrackingParticleKeys.insert(trackingParticleRef.key());
  }

  // initialize a couple of counters
  int count_associatedRecHits{0}, count_RecHitsInSimDoublets{0};

  // loop over pixel RecHit collections of the different pixel modules
  for (const auto& detSet : *hits) {
    // determine layer Id
    unsigned int layerId = simdoublets::getLayerId<TrackerTraits>(detSet.detId(), trackerTopology_);

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
            SiPixelRecHitRef hitRef = edmNew::makeRefTo(hits, &hit);
            count_associatedRecHits++;
            // loop over collection of SimDoublets and find the one of the associated TrackingParticle
            for (auto& simDoublets : simDoubletsCollection) {
              TrackingParticleRef trackingParticleRef = simDoublets.trackingParticle();
              if (assocTrackingParticle.key() == trackingParticleRef.key()) {
                simDoublets.addRecHit(hitRef, layerId);
                count_RecHitsInSimDoublets++;
              }
            }
          }
        }
      }
    }  // end loop over RecHits
  }  // end loop over pixel RecHit collections of the different pixel modules

  // loop over collection of SimDoublets and sort the RecHits according to their position
  for (auto& simDoublets : simDoubletsCollection) {
    simDoublets.sortRecHits();
  }

  LogDebug("SimDoubletsProducer") << "Size of SiPixelRecHitCollection : " << hits->size() << std::endl;
  LogDebug("SimDoubletsProducer") << count_associatedRecHits << " of " << hits->size()
                                  << " RecHits are associated to selected TrackingParticles ("
                                  << count_RecHitsInSimDoublets - count_associatedRecHits
                                  << " of them were associated multiple times)." << std::endl;
  LogDebug("SimDoubletsProducer") << "Number of selected TrackingParticles : " << simDoubletsCollection.size()
                                  << std::endl;
  LogDebug("SimDoubletsProducer") << "Size of TrackingParticle Collection  : " << trackingParticles->size()
                                  << std::endl;

  // put the produced SimDoublets collection in the event
  event.emplace(simDoublets_putToken_, std::move(simDoubletsCollection));
}

// define two plugins for Phase 1 and 2
using SimDoubletsProducerPhase1 = SimDoubletsProducer<pixelTopology::Phase1>;
using SimDoubletsProducerPhase2 = SimDoubletsProducer<pixelTopology::Phase2>;

DEFINE_FWK_MODULE(SimDoubletsProducerPhase1);
DEFINE_FWK_MODULE(SimDoubletsProducerPhase2);