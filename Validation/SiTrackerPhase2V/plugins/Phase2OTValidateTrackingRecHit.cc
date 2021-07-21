// Package:    Phase2OTValidateTrackingRecHit
// Class:      Phase2OTValidateTrackingRecHit
//
/**\class Phase2OTValidateTrackingRecHit Phase2OTValidateTrackingRecHit.cc 
 Description:  Standalone  Plugin for Phase2 RecHit validation
*/
//
// Author: Suvankar Roy Chowdhury
// Date: March 2021
//
// system include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "Validation/SiTrackerPhase2V/interface/Phase2OTValidateRecHitBase.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"

class Phase2OTValidateTrackingRecHit : public Phase2OTValidateRecHitBase {
public:
  explicit Phase2OTValidateTrackingRecHit(const edm::ParameterSet&);
  ~Phase2OTValidateTrackingRecHit() override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void fillOTHistos(const edm::Event& iEvent,
                    const TrackerHitAssociator& associateRecHit,
                    const std::vector<edm::Handle<edm::PSimHitContainer>>& simHits,
                    const std::map<unsigned int, SimTrack>& selectedSimTrackMap);

  edm::ParameterSet config_;
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  const double simtrackminpt_;
  const edm::EDGetTokenT<reco::TrackCollection> tokenTracks_;
  const edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHitTokens_;
};

//
// constructors
//
Phase2OTValidateTrackingRecHit::Phase2OTValidateTrackingRecHit(const edm::ParameterSet& iConfig)
    : Phase2OTValidateRecHitBase(iConfig),
      config_(iConfig),
      trackerHitAssociatorConfig_(iConfig, consumesCollector()),
      simtrackminpt_(iConfig.getParameter<double>("SimTrackMinPt")),
      tokenTracks_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracksSrc"))),
      simTracksToken_(consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("simTracksSrc"))) {
  edm::LogInfo("Phase2OTValidateTrackingRecHit") << ">>> Construct Phase2OTValidateTrackingRecHit ";
  for (const auto& itag : config_.getParameter<std::vector<edm::InputTag>>("PSimHitSource"))
    simHitTokens_.push_back(consumes<edm::PSimHitContainer>(itag));
}

//
// destructor
//
Phase2OTValidateTrackingRecHit::~Phase2OTValidateTrackingRecHit() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  edm::LogInfo("Phase2OTValidateTrackingRecHit") << ">>> Destroy Phase2OTValidateTrackingRecHit ";
}

//
// -- Analyze
//
void Phase2OTValidateTrackingRecHit::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
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
  for (edm::SimTrackContainer::const_iterator simTrackIt(simTracks->begin()); simTrackIt != simTracks->end();
       ++simTrackIt) {
    if (simTrackIt->momentum().pt() > simtrackminpt_) {
      selectedSimTrackMap.insert(std::make_pair(simTrackIt->trackId(), *simTrackIt));
    }
  }
  TrackerHitAssociator associateRecHit(iEvent, trackerHitAssociatorConfig_);
  fillOTHistos(iEvent, associateRecHit, simHits, selectedSimTrackMap);
}

void Phase2OTValidateTrackingRecHit::fillOTHistos(const edm::Event& iEvent,
                                                  const TrackerHitAssociator& associateRecHit,
                                                  const std::vector<edm::Handle<edm::PSimHitContainer>>& simHits,
                                                  const std::map<unsigned int, SimTrack>& selectedSimTrackMap) {
  const auto& tracks = iEvent.getHandle(tokenTracks_);
  if (!tracks.isValid())
    return;
  std::map<std::string, unsigned int> nrechitLayerMapP_primary;
  std::map<std::string, unsigned int> nrechitLayerMapS_primary;
  // Loop over tracks
  for (const auto& track : *tracks) {
    for (const auto& hit : track.recHits()) {
      // Get the detector unit's id
      if (!hit->isValid())
        continue;
      auto detId = hit->geographicalId();
      // check that we are in the pixel
      auto subdetid = (detId.subdetId());
      if (subdetid == PixelSubdetector::PixelBarrel || subdetid == PixelSubdetector::PixelEndcap)
        continue;

      // determine the detector we are in
      TrackerGeometry::ModuleType mType = tkGeom_->getDetectorType(detId);
      std::string key = phase2tkutil::getOTHistoId(detId.rawId(), tTopo_);
      if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
        if (nrechitLayerMapP_primary.find(key) == nrechitLayerMapP_primary.end()) {
          nrechitLayerMapP_primary.emplace(key, 1);
        } else {
          nrechitLayerMapP_primary[key] += 1;
        }
      } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
        if (nrechitLayerMapS_primary.find(key) == nrechitLayerMapS_primary.end()) {
          nrechitLayerMapS_primary.emplace(key, 1);
        } else {
          nrechitLayerMapS_primary[key] += 1;
        }
      }
      //GetSimHits
      const Phase2TrackerRecHit1D* rechit = dynamic_cast<const Phase2TrackerRecHit1D*>(hit);
      if (!rechit) {
        edm::LogError("Phase2OTValidateTrackingRecHit")
            << "Cannot cast tracking rechit to Phase2TrackerRecHit1D!" << std::endl;
        continue;
      }
      const std::vector<SimHitIdpr>& matchedId = associateRecHit.associateHitId(*rechit);
      const PSimHit* simhitClosest = nullptr;
      LocalPoint lp = rechit->localPosition();
      float mind = 1e4;
      for (const auto& simHitCol : simHits) {
        for (const auto& simhitIt : *simHitCol) {
          if (detId.rawId() != simhitIt.detUnitId())
            continue;
          for (auto& mId : matchedId) {
            if (simhitIt.trackId() == mId.first) {
              float dx = simhitIt.localPosition().x() - lp.x();
              float dy = simhitIt.localPosition().y() - lp.y();
              float dist = dx * dx + dy * dy;
              if (!simhitClosest || dist < mind) {
                mind = dist;
                simhitClosest = &simhitIt;
              }
            }
          }
        }  //end loop over PSimhitcontainers
      }    //end loop over simHits
      if (!simhitClosest)
        continue;
      fillOTRecHitHistos(
          simhitClosest, rechit, selectedSimTrackMap, nrechitLayerMapP_primary, nrechitLayerMapS_primary);

    }  //end loop over rechits of a track
  }    //End loop over tracks

  //fill nRecHits per event
  //fill nRecHit counter per layer
  for (auto& lme : nrechitLayerMapP_primary) {
    layerMEs_[lme.first].numberRecHitsprimary_P->Fill(nrechitLayerMapP_primary[lme.first]);
  }
  for (auto& lme : nrechitLayerMapS_primary) {
    layerMEs_[lme.first].numberRecHitsprimary_S->Fill(nrechitLayerMapS_primary[lme.first]);
  }
}

void Phase2OTValidateTrackingRecHit::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // call the base fillPsetDescription for the plots bookings
  Phase2OTValidateRecHitBase::fillPSetDescription(desc);

  //for macro-pixel sensors
  ///////
  desc.add<edm::InputTag>("SimVertexSource", edm::InputTag("g4SimHits"));
  desc.add<bool>("associatePixel", false);
  desc.add<bool>("associateHitbySimTrack", true);
  desc.add<bool>("Verbosity", false);
  desc.add<bool>("associateStrip", true);
  desc.add<edm::InputTag>("phase2TrackerSimLinkSrc", edm::InputTag("simSiPixelDigis", "Tracker"));
  desc.add<bool>("associateRecoTracks", false);
  desc.add<edm::InputTag>("pixelSimLinkSrc", edm::InputTag("simSiPixelDigis", "Pixel"));
  desc.add<bool>("usePhase2Tracker", true);
  desc.add<edm::InputTag>("rechitsSrc", edm::InputTag("siPhase2RecHits"));
  desc.add<edm::InputTag>("simTracksSrc", edm::InputTag("g4SimHits"));
  desc.add<double>("SimTrackMinPt", 2.0);
  desc.add<std::vector<edm::InputTag>>("PSimHitSource",
                                       {
                                           edm::InputTag("g4SimHits:TrackerHitsTIBLowTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsTIBHighTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsTIDLowTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsTIDHighTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsTOBLowTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsTOBHighTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsTECLowTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsTECHighTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsPixelBarrelLowTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsPixelBarrelHighTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsPixelEndcapLowTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsPixelEndcapHighTof"),
                                       });

  desc.add<std::vector<std::string>>("ROUList",
                                     {"TrackerHitsPixelBarrelLowTof",
                                      "TrackerHitsPixelBarrelHighTof",
                                      "TrackerHitsTIBLowTof",
                                      "TrackerHitsTIBHighTof",
                                      "TrackerHitsTIDLowTof",
                                      "TrackerHitsTIDHighTof",
                                      "TrackerHitsTOBLowTof",
                                      "TrackerHitsTOBHighTof",
                                      "TrackerHitsPixelEndcapLowTof",
                                      "TrackerHitsPixelEndcapHighTof",
                                      "TrackerHitsTECLowTof",
                                      "TrackerHitsTECHighTof"});
  desc.add<edm::InputTag>("tracksSrc", edm::InputTag("generalTracks"));
  desc.add<std::string>("TopFolderName", "TrackerPhase2OTTrackingRecHitV");
  descriptions.add("Phase2OTValidateTrackingRecHit", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2OTValidateTrackingRecHit);
