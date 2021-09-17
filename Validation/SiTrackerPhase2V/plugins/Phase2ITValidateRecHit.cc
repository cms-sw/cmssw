// Package:    Phase2ITValidateRecHit
// Class:      Phase2ITValidateRecHit
//
/**\class Phase2ITValidateRecHit Phase2ITValidateRecHit.cc 
 Description:  Plugin for Phase2 RecHit validation
*/
//
// Author: Shubhi Parolia, Suvankar Roy Chowdhury
// Date: June 2020
//
// system include files
#include <memory>
#include <map>
#include <vector>
#include <algorithm>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
//--- for SimHit association
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
// base class
#include "Validation/SiTrackerPhase2V/interface/Phase2ITValidateRecHitBase.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"

class Phase2ITValidateRecHit : public Phase2ITValidateRecHitBase {
public:
  explicit Phase2ITValidateRecHit(const edm::ParameterSet&);
  ~Phase2ITValidateRecHit() override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void fillITHistos(const edm::Event& iEvent,
                    const TrackerHitAssociator& associateRecHit,
                    const std::vector<edm::Handle<edm::PSimHitContainer>>& simHits,
                    const std::map<unsigned int, SimTrack>& selectedSimTrackMap);

  edm::ParameterSet config_;
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  const double simtrackminpt_;
  const edm::EDGetTokenT<SiPixelRecHitCollection> tokenRecHitsIT_;
  const edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHitTokens_;
};

Phase2ITValidateRecHit::Phase2ITValidateRecHit(const edm::ParameterSet& iConfig)
    : Phase2ITValidateRecHitBase(iConfig),
      config_(iConfig),
      trackerHitAssociatorConfig_(iConfig, consumesCollector()),
      simtrackminpt_(iConfig.getParameter<double>("SimTrackMinPt")),
      tokenRecHitsIT_(consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("rechitsSrc"))),
      simTracksToken_(consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("simTracksSrc"))) {
  edm::LogInfo("Phase2ITValidateRecHit") << ">>> Construct Phase2ITValidateRecHit ";
  for (const auto& itName : config_.getParameter<std::vector<std::string>>("ROUList")) {
    simHitTokens_.push_back(consumes<std::vector<PSimHit>>(edm::InputTag("g4SimHits", itName)));
  }
}
//
Phase2ITValidateRecHit::~Phase2ITValidateRecHit() {
  edm::LogInfo("Phase2ITValidateRecHit") << ">>> Destroy Phase2ITValidateRecHit ";
}

void Phase2ITValidateRecHit::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::vector<edm::Handle<edm::PSimHitContainer>> simHits;
  for (const auto& itoken : simHitTokens_) {
    const auto& simHitHandle = iEvent.getHandle(itoken);
    if (!simHitHandle.isValid())
      continue;
    simHits.emplace_back(simHitHandle);
  }
  // Get the SimTracks and push them in a map of id, SimTrack
  const auto& simTracks = iEvent.getHandle(simTracksToken_);

  std::map<unsigned int, SimTrack> selectedSimTrackMap;
  for (const auto& simTrackIt : *simTracks) {
    if (simTrackIt.momentum().pt() > simtrackminpt_) {
      selectedSimTrackMap.emplace(simTrackIt.trackId(), simTrackIt);
    }
  }
  TrackerHitAssociator associateRecHit(iEvent, trackerHitAssociatorConfig_);
  fillITHistos(iEvent, associateRecHit, simHits, selectedSimTrackMap);
}

void Phase2ITValidateRecHit::fillITHistos(const edm::Event& iEvent,
                                          const TrackerHitAssociator& associateRecHit,
                                          const std::vector<edm::Handle<edm::PSimHitContainer>>& simHits,
                                          const std::map<unsigned int, SimTrack>& selectedSimTrackMap) {
  // Get the RecHits
  const auto& rechits = iEvent.getHandle(tokenRecHitsIT_);
  if (!rechits.isValid())
    return;
  std::map<std::string, unsigned int> nrechitLayerMap_primary;
  // Loop over modules
  for (const auto& DSViter : *rechits) {
    // Get the detector unit's id
    unsigned int rawid(DSViter.detId());
    DetId detId(rawid);
    // determine the detector we are in
    std::string key = phase2tkutil::getITHistoId(detId.rawId(), tTopo_);
    if (nrechitLayerMap_primary.find(key) == nrechitLayerMap_primary.end()) {
      nrechitLayerMap_primary.emplace(key, DSViter.size());
    } else {
      nrechitLayerMap_primary[key] += DSViter.size();
    }
    //loop over rechits for a single detId
    for (const auto& rechit : DSViter) {
      //GetSimHits
      const std::vector<SimHitIdpr>& matchedId = associateRecHit.associateHitId(rechit);
      const PSimHit* simhitClosest = nullptr;
      float minx = 10000;
      LocalPoint lp = rechit.localPosition();
      for (const auto& simHitCol : simHits) {
        for (const auto& simhitIt : *simHitCol) {
          if (detId.rawId() != simhitIt.detUnitId())
            continue;
          for (const auto& mId : matchedId) {
            if (simhitIt.trackId() == mId.first) {
              if (!simhitClosest || abs(simhitIt.localPosition().x() - lp.x()) < minx) {
                minx = std::abs(simhitIt.localPosition().x() - lp.x());
                simhitClosest = &simhitIt;
              }
            }
          }
        }  //end loop over PSimhitcontainers
      }    //end loop over simHits
      if (!simhitClosest)
        continue;

      // call the base class method to fill the plots
      fillRechitHistos(simhitClosest, &rechit, selectedSimTrackMap, nrechitLayerMap_primary);

    }  //end loop over rechits of a detId
  }    //End loop over DetSetVector

  //fill nRecHit counter per layer
  for (const auto& lme : nrechitLayerMap_primary) {
    layerMEs_[lme.first].numberRecHitsprimary->Fill(nrechitLayerMap_primary[lme.first]);
  }
}

void Phase2ITValidateRecHit::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // rechitValidIT
  edm::ParameterSetDescription desc;

  // call the base fillPsetDescription for the plots bookings
  Phase2ITValidateRecHitBase::fillPSetDescription(desc);

  //to be used in TrackerHitAssociator
  desc.add<bool>("associatePixel", true);
  desc.add<bool>("associateStrip", false);
  desc.add<bool>("usePhase2Tracker", true);
  desc.add<bool>("associateRecoTracks", false);
  desc.add<bool>("associateHitbySimTrack", true);
  desc.add<edm::InputTag>("pixelSimLinkSrc", edm::InputTag("simSiPixelDigis", "Pixel"));
  desc.add<std::vector<std::string>>("ROUList",
                                     {
                                         "TrackerHitsPixelBarrelLowTof",
                                         "TrackerHitsPixelBarrelHighTof",
                                         "TrackerHitsPixelEndcapLowTof",
                                         "TrackerHitsPixelEndcapHighTof",
                                     });
  //
  desc.add<edm::InputTag>("simTracksSrc", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("SimVertexSource", edm::InputTag("g4SimHits"));
  desc.add<double>("SimTrackMinPt", 2.0);
  desc.add<edm::InputTag>("rechitsSrc", edm::InputTag("siPixelRecHits"));
  desc.add<std::string>("TopFolderName", "TrackerPhase2ITRecHitV");
  desc.add<bool>("Verbosity", false);
  descriptions.add("Phase2ITValidateRecHit", desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(Phase2ITValidateRecHit);
