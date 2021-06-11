// Package:    Phase2OTValidateRecHit
// Class:      Phase2OTValidateRecHit
//
/**\class Phase2OTValidateRecHit Phase2OTValidateRecHit.cc 
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

class Phase2OTValidateRecHit : public Phase2OTValidateRecHitBase {
public:
  explicit Phase2OTValidateRecHit(const edm::ParameterSet&);
  ~Phase2OTValidateRecHit() override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void fillOTHistos(const edm::Event& iEvent,
                    const TrackerHitAssociator& associateRecHit,
                    const std::vector<edm::Handle<edm::PSimHitContainer>>& simHits,
                    const std::map<unsigned int, SimTrack>& selectedSimTrackMap);

  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  const double simtrackminpt_;
  const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> tokenRecHitsOT_;
  const edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHitTokens_;
};

//
// constructors
//
Phase2OTValidateRecHit::Phase2OTValidateRecHit(const edm::ParameterSet& iConfig)
    : Phase2OTValidateRecHitBase(iConfig),
      trackerHitAssociatorConfig_(iConfig, consumesCollector()),
      simtrackminpt_(iConfig.getParameter<double>("SimTrackMinPt")),
      tokenRecHitsOT_(consumes<Phase2TrackerRecHit1DCollectionNew>(iConfig.getParameter<edm::InputTag>("rechitsSrc"))),
      simTracksToken_(consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("simTracksSrc"))) {
  edm::LogInfo("Phase2OTValidateRecHit") << ">>> Construct Phase2OTValidateRecHit ";
  for (const auto& itag : config_.getParameter<std::vector<edm::InputTag>>("PSimHitSource"))
    simHitTokens_.push_back(consumes<edm::PSimHitContainer>(itag));
}

//
// destructor
//
Phase2OTValidateRecHit::~Phase2OTValidateRecHit() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  edm::LogInfo("Phase2OTValidateRecHit") << ">>> Destroy Phase2OTValidateRecHit ";
}

//
// -- Analyze
//
void Phase2OTValidateRecHit::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
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
  for (const auto& simTrackIt : *simTracks)
    if (simTrackIt.momentum().pt() > simtrackminpt_) {
      selectedSimTrackMap.insert(std::make_pair(simTrackIt.trackId(), simTrackIt));
    }
  TrackerHitAssociator associateRecHit(iEvent, trackerHitAssociatorConfig_);
  fillOTHistos(iEvent, associateRecHit, simHits, selectedSimTrackMap);
}

void Phase2OTValidateRecHit::fillOTHistos(const edm::Event& iEvent,
                                          const TrackerHitAssociator& associateRecHit,
                                          const std::vector<edm::Handle<edm::PSimHitContainer>>& simHits,
                                          const std::map<unsigned int, SimTrack>& selectedSimTrackMap) {
  // Get the RecHits
  const auto& rechits = iEvent.getHandle(tokenRecHitsOT_);
  if (!rechits.isValid())
    return;
  std::map<std::string, unsigned int> nrechitLayerMapP_primary;
  std::map<std::string, unsigned int> nrechitLayerMapS_primary;
  // Loop over modules
  Phase2TrackerRecHit1DCollectionNew::const_iterator DSViter;
  for (const auto& DSViter : *rechits) {
    // Get the detector unit's id
    unsigned int rawid(DSViter.detId());
    DetId detId(rawid);
    // determine the detector we are in
    TrackerGeometry::ModuleType mType = tkGeom_->getDetectorType(detId);
    std::string key = phase2tkutil::getOTHistoId(detId.rawId(), tTopo_);
    if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
      if (nrechitLayerMapP_primary.find(key) == nrechitLayerMapP_primary.end()) {
        nrechitLayerMapP_primary.insert(std::make_pair(key, DSViter.size()));
      } else {
        nrechitLayerMapP_primary[key] += DSViter.size();
      }
    } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
      if (nrechitLayerMapS_primary.find(key) == nrechitLayerMapS_primary.end()) {
        nrechitLayerMapS_primary.insert(std::make_pair(key, DSViter.size()));
      } else {
        nrechitLayerMapS_primary[key] += DSViter.size();
      }
    }
    //loop over rechits for a single detId
    for (const auto& rechit : DSViter) {
      //GetSimHits
      const std::vector<SimHitIdpr>& matchedId = associateRecHit.associateHitId(rechit);
      const PSimHit* simhitClosest = nullptr;
      LocalPoint lp = rechit.localPosition();
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
          simhitClosest, &rechit, selectedSimTrackMap, nrechitLayerMapP_primary, nrechitLayerMapS_primary);

    }  //end loop over rechits of a detId
  }    //End loop over DetSetVector

  //fill nRecHits per event
  //fill nRecHit counter per layer
  for (auto& lme : nrechitLayerMapP_primary) {
    layerMEs_[lme.first].numberRecHitsprimary_P->Fill(nrechitLayerMapP_primary[lme.first]);
  }
  for (auto& lme : nrechitLayerMapS_primary) {
    layerMEs_[lme.first].numberRecHitsprimary_S->Fill(nrechitLayerMapS_primary[lme.first]);
  }
}

void Phase2OTValidateRecHit::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // call the base fillPsetDescription for the plots bookings
  Phase2OTValidateRecHitBase::fillPSetDescription(desc);

  //for macro-pixel sensors
  ///////
  desc.add<edm::InputTag>("SimVertexSource", edm::InputTag("g4SimHits"));
  desc.add<bool>("associatePixel", false);
  desc.add<std::string>("TopFolderName", "TrackerPhase2OTRecHitV");
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

  descriptions.add("Phase2OTValidateRecHit", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2OTValidateRecHit);
