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
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
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
  pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const* __restrict__ cpeParams_ = nullptr;
  const TrackerTopology* trackerTopology_ = nullptr;
  const TrackerGeometry* trackerGeometry_ = nullptr;
  unsigned int numLayersOT_ = 2;

  // tokens for ClusterParameterEstimator, tracker topology, ect.
  const edm::ESGetToken<PixelCPEFastParamsHost<TrackerTraits>, PixelCPEFastParamsRecord> cpe_getToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topology_getToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometry_getToken_;
  const edm::EDGetTokenT<ClusterTPAssociation> clusterTPAssociation_getToken_;
  const edm::EDGetTokenT<TrackingParticleCollection> trackingParticles_getToken_;
  const edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHits_getToken_;
  const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> otRecHits_getToken_;
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
    }
    // return the determined Id
    return layerId;
  }

  // function that determines the cluster size of a RecHit in local y direction
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
SimDoubletsProducer<TrackerTraits>::SimDoubletsProducer(const edm::ParameterSet& pSet)
    : cpe_getToken_(esConsumes(edm::ESInputTag("", pSet.getParameter<std::string>("CPE")))),
      topology_getToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      geometry_getToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      clusterTPAssociation_getToken_(
          consumes<ClusterTPAssociation>(pSet.getParameter<edm::InputTag>("clusterTPAssociationSrc"))),
      trackingParticles_getToken_(consumes(pSet.getParameter<edm::InputTag>("trackingParticleSrc"))),
      pixelRecHits_getToken_(consumes(pSet.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      otRecHits_getToken_(consumes(pSet.getParameter<edm::InputTag>("outerTrackerRecHitSrc"))),
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
  if (!hitsOT.isValid()) {
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

  // initialize a couple of variables used in the following loop
  unsigned int detId, layerId, layerIdOT, maxCol;
  uint16_t pixmx;
  int moduleId, clusterYSize;

  // loop over pixel RecHit collections of the different pixel modules
  for (const auto& detSet : *hits) {
    // get detector Id
    detId = detSet.detId();
    DetId detIdObject(detId);

    // determine layer Id from detector Id
    layerId = simdoublets::getLayerId<TrackerTraits>(detId, trackerTopology_);

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
            SiPixelRecHitRef hitRef = edmNew::makeRefTo(hits, &hit);
            clusterYSize = simdoublets::clusterYSize(hit.cluster(), pixmx, maxCol);
            count_associatedRecHits++;
            // loop over collection of SimDoublets and find the one of the associated TrackingParticle
            for (auto& simDoublets : simDoubletsCollection) {
              TrackingParticleRef trackingParticleRef = simDoublets.trackingParticle();
              if (assocTrackingParticle.key() == trackingParticleRef.key()) {
                simDoublets.addRecHit(hitRef, layerId, clusterYSize, detId, moduleId);
                count_RecHitsInSimDoublets++;
              }
            }
          }
        }
      }
    }  // end loop over RecHits
  }  // end loop over pixel RecHit collections of the different pixel modules

  // loop over Outer Tracker RecHit collections of the different modules
  for (const auto& detSetOT : *hitsOT) {
    // get detector Id
    detId = detSetOT.detId();

    // get layerId of the OT
    layerIdOT = trackerTopology_->getOTLayerNumber(detId);

    // only use the RecHits if the module is in the accepted range of layers and one of lower modules
    if ((layerIdOT <= numLayersOT_) && trackerTopology_->isLower(detId)) {
      // determine layer Id from detector Id plus the offset from the pixel layers:
      // layerId = layerId(OT) + N(pixelLayers) - 1
      // the (-1) comes from the layerId(OT) starting from 1 instead of 0
      layerId = layerIdOT + TrackerTraits::numberOfLayers - 1;

      // loop over RecHits
      for (auto const& hitOT : detSetOT) {
        std::cout << "OT RecHit in layer " << layerId << ": " << hitOT.globalPosition() << std::endl;
      }  // end loop over RecHits
    }
  }  // end loop over OT RecHit collections of the different modules

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