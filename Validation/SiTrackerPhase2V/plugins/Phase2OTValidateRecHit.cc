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
#include <memory>
#include <map>
#include "FWCore/Framework/interface/Frameworkfwd.h"
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
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2ValidationUtil.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"

class Phase2OTValidateRecHit : public DQMEDAnalyzer {
public:
  explicit Phase2OTValidateRecHit(const edm::ParameterSet&);
  ~Phase2OTValidateRecHit() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void fillOTHistos(const edm::Event& iEvent,
                    const TrackerHitAssociator& associateRecHit,
                    const std::vector<edm::Handle<edm::PSimHitContainer>>& simHits,
                    const std::map<unsigned int, SimTrack>& selectedSimTrackMap);

  void bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir);

  edm::ParameterSet config_;
  bool pixelFlag_;
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  const double simtrackminpt_;
  std::string geomType_;
  const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> tokenRecHitsOT_;
  const edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHitTokens_;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;

  struct RecHitME {
    // use TH1D instead of TH1F to avoid stauration at 2^31
    // above this increments with +1 don't work for float, need double
    MonitorElement* deltaX_P = nullptr;
    MonitorElement* deltaX_S = nullptr;
    MonitorElement* deltaY_P = nullptr;
    MonitorElement* deltaY_S = nullptr;
    MonitorElement* pullX_P = nullptr;
    MonitorElement* pullX_S = nullptr;
    MonitorElement* pullY_P = nullptr;
    MonitorElement* pullY_S = nullptr;
    MonitorElement* deltaX_eta_P = nullptr;
    MonitorElement* deltaX_eta_S = nullptr;
    MonitorElement* deltaY_eta_P = nullptr;
    MonitorElement* deltaY_eta_S = nullptr;
    MonitorElement* pullX_eta_P = nullptr;
    MonitorElement* pullX_eta_S = nullptr;
    MonitorElement* pullY_eta_P = nullptr;
    MonitorElement* pullY_eta_S = nullptr;
    //For rechits matched to simhits from highPT tracks
    MonitorElement* pullX_primary_P;
    MonitorElement* pullX_primary_S;
    MonitorElement* pullY_primary_P;
    MonitorElement* pullY_primary_S;
    MonitorElement* deltaX_primary_P;
    MonitorElement* deltaX_primary_S;
    MonitorElement* deltaY_primary_P;
    MonitorElement* deltaY_primary_S;
    MonitorElement* numberRecHitsprimary_P;
    MonitorElement* numberRecHitsprimary_S;
  };
  std::map<std::string, RecHitME> layerMEs_;
};

//
// constructors
//
Phase2OTValidateRecHit::Phase2OTValidateRecHit(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      trackerHitAssociatorConfig_(iConfig, consumesCollector()),
      simtrackminpt_(iConfig.getParameter<double>("SimTrackMinPt")),
      tokenRecHitsOT_(consumes<Phase2TrackerRecHit1DCollectionNew>(config_.getParameter<edm::InputTag>("rechitsSrc"))),
      simTracksToken_(consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("simTracksSrc"))),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
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
// -- DQM Begin Run
void Phase2OTValidateRecHit::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
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
  for (edm::SimTrackContainer::const_iterator simTrackIt(simTracks->begin()); simTrackIt != simTracks->end();
       ++simTrackIt) {
    if (simTrackIt->momentum().pt() > simtrackminpt_) {
      selectedSimTrackMap.insert(std::make_pair(simTrackIt->trackId(), *simTrackIt));
    }
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
  unsigned long int nTotrechitsinevt = 0;
  // Loop over modules
  Phase2TrackerRecHit1DCollectionNew::const_iterator DSViter;
  for (DSViter = rechits->begin(); DSViter != rechits->end(); ++DSViter) {
    // Get the detector unit's id
    unsigned int rawid(DSViter->detId());
    DetId detId(rawid);
    // Get the geomdet
    const GeomDetUnit* geomDetunit(tkGeom_->idToDetUnit(detId));
    if (!geomDetunit)
      continue;
    // determine the detector we are in
    TrackerGeometry::ModuleType mType = tkGeom_->getDetectorType(detId);
    std::string key = phase2tkutil::getOTHistoId(detId.rawId(), tTopo_);
    nTotrechitsinevt += DSViter->size();
    if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
      if (nrechitLayerMapP_primary.find(key) == nrechitLayerMapP_primary.end()) {
        nrechitLayerMapP_primary.insert(std::make_pair(key, DSViter->size()));
      } else {
        nrechitLayerMapP_primary[key] += DSViter->size();
      }
    } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
      if (nrechitLayerMapS_primary.find(key) == nrechitLayerMapS_primary.end()) {
        nrechitLayerMapS_primary.insert(std::make_pair(key, DSViter->size()));
      } else {
        nrechitLayerMapS_primary[key] += DSViter->size();
      }
    }
    edmNew::DetSet<Phase2TrackerRecHit1D>::const_iterator rechitIt;
    //loop over rechits for a single detId
    for (rechitIt = DSViter->begin(); rechitIt != DSViter->end(); ++rechitIt) {
      LocalPoint lp = rechitIt->localPosition();
      //GetSimHits
      const std::vector<SimHitIdpr>& matchedId = associateRecHit.associateHitId(*rechitIt);
      const PSimHit* simhitClosest = nullptr;
      float mind = 1e4;
      for (unsigned int si = 0; si < simHits.size(); ++si) {
        for (edm::PSimHitContainer::const_iterator simhitIt = simHits.at(si)->begin();
             simhitIt != simHits.at(si)->end();
             ++simhitIt) {
          if (detId.rawId() != simhitIt->detUnitId())
            continue;
          for (auto& mId : matchedId) {
            if (simhitIt->trackId() == mId.first) {
              float dx = simhitIt->localPosition().x() - lp.x();
              float dy = simhitIt->localPosition().y() - lp.y();
              float dist = std::sqrt(dx * dx + dy * dy);
              if (!simhitClosest || dist < mind) {
                mind = dist;
                simhitClosest = &*simhitIt;
              }
            }
          }
        }  //end loop over PSimhitcontainers
      }    //end loop over simHits
      if (!simhitClosest)
        continue;
      auto simTrackIt(selectedSimTrackMap.find(simhitClosest->trackId()));
      bool isPrimary = false;
      //check if simhit is primary
      if (simTrackIt != selectedSimTrackMap.end())
        isPrimary = phase2tkutil::isPrimary(simTrackIt->second, simhitClosest);
      Local3DPoint simlp(simhitClosest->localPosition());
      const LocalError& lperr = rechitIt->localPositionError();
      double dx = lp.x() - simlp.x();
      double dy = lp.y() - simlp.y();
      double pullx = 999.;
      double pully = 999.;
      if (lperr.xx())
        pullx = (lp.x() - simlp.x()) / std::sqrt(lperr.xx());
      if (lperr.yy())
        pully = (lp.y() - simlp.y()) / std::sqrt(lperr.yy());
      float eta = geomDetunit->surface().toGlobal(lp).eta();
      ;
      if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
        layerMEs_[key].deltaX_P->Fill(phase2tkutil::cmtomicron * dx);
        layerMEs_[key].deltaY_P->Fill(phase2tkutil::cmtomicron * dy);
        layerMEs_[key].pullX_P->Fill(pullx);
        layerMEs_[key].pullY_P->Fill(pully);
        layerMEs_[key].deltaX_eta_P->Fill(eta, phase2tkutil::cmtomicron * dx);
        layerMEs_[key].deltaY_eta_P->Fill(eta, phase2tkutil::cmtomicron * dy);
        layerMEs_[key].pullX_eta_P->Fill(eta, pullx);
        layerMEs_[key].pullY_eta_P->Fill(eta, pully);
        if (isPrimary) {
          layerMEs_[key].deltaX_primary_P->Fill(phase2tkutil::cmtomicron * dx);
          layerMEs_[key].deltaY_primary_P->Fill(phase2tkutil::cmtomicron * dy);
          layerMEs_[key].pullX_primary_P->Fill(pullx);
          layerMEs_[key].pullY_primary_P->Fill(pully);
        } else
          nrechitLayerMapP_primary[key]--;
      } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
        layerMEs_[key].deltaX_S->Fill(phase2tkutil::cmtomicron * dx);
        layerMEs_[key].deltaY_S->Fill(dy);
        layerMEs_[key].pullX_S->Fill(pullx);
        layerMEs_[key].pullY_S->Fill(pully);
        layerMEs_[key].deltaX_eta_S->Fill(eta, phase2tkutil::cmtomicron * dx);
        layerMEs_[key].deltaY_eta_S->Fill(eta, dy);
        layerMEs_[key].pullX_eta_S->Fill(eta, pullx);
        layerMEs_[key].pullY_eta_S->Fill(eta, pully);
        if (isPrimary) {
          layerMEs_[key].deltaX_primary_S->Fill(phase2tkutil::cmtomicron * dx);
          layerMEs_[key].deltaY_primary_S->Fill(dy);
          layerMEs_[key].pullX_primary_S->Fill(pullx);
          layerMEs_[key].pullY_primary_S->Fill(pully);
        } else
          nrechitLayerMapS_primary[key]--;
      }
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
//
// -- Book Histograms
//
void Phase2OTValidateRecHit::bookHistograms(DQMStore::IBooker& ibooker,
                                            edm::Run const& iRun,
                                            edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  //Now book layer wise histos
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(iSetup)) {
    for (auto const& det_u : tkGeom_->detUnits()) {
      //Always check TrackerNumberingBuilder before changing this part
      if (det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXB ||
          det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC)
        continue;
      unsigned int detId_raw = det_u->geographicalId().rawId();
      bookLayerHistos(ibooker, detId_raw, top_folder);
    }
  }
}

//
// -- Book Layer Histograms
//
void Phase2OTValidateRecHit::bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir) {
  std::string key = phase2tkutil::getOTHistoId(det_id, tTopo_);
  if (layerMEs_.find(key) == layerMEs_.end()) {
    ibooker.cd();
    RecHitME local_histos;
    ibooker.setCurrentFolder(subdir + "/" + key);
    edm::LogInfo("Phase2OTValidateRecHit") << " Booking Histograms in : " << key;

    if (tkGeom_->getDetectorType(det_id) == TrackerGeometry::ModuleType::Ph2PSP) {
      local_histos.deltaX_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_Pixel"), ibooker);
      local_histos.deltaY_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_Y_Pixel"), ibooker);

      local_histos.pullX_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Pull_X_Pixel"), ibooker);
      local_histos.pullY_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Pull_X_Pixel"), ibooker);

      local_histos.deltaX_eta_P =
          phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_vs_eta_Pixel"), ibooker);
      local_histos.deltaY_eta_P =
          phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_Y_vs_eta_Pixel"), ibooker);

      local_histos.pullX_eta_P =
          phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("Pull_X_vs_eta_Pixel"), ibooker);
      local_histos.pullY_eta_P =
          phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("Pull_X_vs_eta_Pixel"), ibooker);

      ibooker.setCurrentFolder(subdir + "/" + key + "/PrimarySimHits");
      //all histos for Primary particles
      local_histos.numberRecHitsprimary_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("nRecHits_Pixel_primary"), ibooker);

      local_histos.deltaX_primary_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_Pixel_Primary"), ibooker);
      local_histos.deltaY_primary_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_Pixel_Primary"), ibooker);

      local_histos.pullX_primary_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Pull_X_Pixel_Primary"), ibooker);
      local_histos.pullY_primary_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Pull_X_Pixel_Primary"), ibooker);
    }  //if block for P

    ibooker.setCurrentFolder(subdir + "/" + key);
    local_histos.deltaX_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_Strip"), ibooker);
    local_histos.deltaY_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_Y_Strip"), ibooker);

    local_histos.pullX_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Pull_X_Strip"), ibooker);
    local_histos.pullY_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Pull_Y_Strip"), ibooker);

    local_histos.deltaX_eta_S =
        phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_vs_eta_Strip"), ibooker);
    local_histos.deltaY_eta_S =
        phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_Y_vs_eta_Strip"), ibooker);

    local_histos.pullX_eta_S =
        phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("Pull_X_vs_eta_Strip"), ibooker);
    local_histos.pullY_eta_S =
        phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("Pull_X_vs_eta_Pixel"), ibooker);

    //primary
    ibooker.setCurrentFolder(subdir + "/" + key + "/PrimarySimHits");
    local_histos.numberRecHitsprimary_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("nRecHits_Strip_primary"), ibooker);

    local_histos.deltaX_primary_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_Strip_Primary"), ibooker);
    local_histos.deltaY_primary_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_Y_Strip_Primary"), ibooker);

    local_histos.pullX_primary_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Pull_X_Strip_Primary"), ibooker);
    local_histos.pullY_primary_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Pull_X_Strip_Primary"), ibooker);

    layerMEs_.insert(std::make_pair(key, local_histos));
  }
}

void Phase2OTValidateRecHit::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // rechitValidOT
  //for macro-pixel sensors
  std::string mptag = "macro-pixel sensor";
  std::string striptag = "strip sensor";
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X_Pixel");
    psd0.add<std::string>("title", "#Delta X " + mptag + ";Cluster resolution X coordinate [#mum]");
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 250);
    psd0.add<double>("xmin", -250);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Delta_X_Pixel", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_Pixel");
    psd0.add<std::string>("title", "#Delta Y " + mptag + ";Cluster resolution Y coordinate [#mum]");
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmin", -1500);
    psd0.add<double>("xmax", 1500);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Delta_Y_Pixel", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X_Pixel_Primary");
    psd0.add<std::string>("title", "#Delta X " + mptag + ";cluster resolution X coordinate [#mum]");
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmin", -250);
    psd0.add<double>("xmax", 250);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Delta_X_Pixel_Primary", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_Pixel_Primary");
    psd0.add<std::string>("title", "#Delta Y " + mptag + ";cluster resolution Y coordinate [#mum]");
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmin", -500);
    psd0.add<double>("xmax", 500);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Delta_Y_Pixel_Primary", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X_vs_Eta_Pixel");
    psd0.add<std::string>("title", ";#eta;#Delta x [#mum]");
    psd0.add<double>("ymin", -250.0);
    psd0.add<double>("ymax", 250.0);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    desc.add<edm::ParameterSetDescription>("Delta_X_vs_eta_Pixel", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_vs_Eta_Pixel");
    psd0.add<std::string>("title", ";#eta;#Delta y [#mum]");
    psd0.add<double>("ymin", -1500.0);
    psd0.add<double>("ymax", 1500.0);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    desc.add<edm::ParameterSetDescription>("Delta_Y_vs_eta_Pixel", psd0);
  }

  //Pulls macro-pixel
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_X_Pixel");
    psd0.add<std::string>("title", ";pull x;");
    psd0.add<double>("xmin", -4.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Pull_X_Pixel", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_Y_Pixel");
    psd0.add<std::string>("title", ";pull y;");
    psd0.add<double>("xmin", -4.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Pull_Y_Pixel", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_X_Pixel_Primary");
    psd0.add<std::string>("title", ";pull x;");
    psd0.add<double>("xmin", -4.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Pull_X_Pixel_Primary", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_Y_Pixel_Primary");
    psd0.add<std::string>("title", ";pull y;");
    psd0.add<double>("xmin", -4.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Pull_Y_Pixel_Primary", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_X_vs_Eta");
    psd0.add<std::string>("title", ";#eta;pull x");
    psd0.add<double>("ymax", 4.0);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    psd0.add<double>("ymin", -4.0);
    desc.add<edm::ParameterSetDescription>("Pull_X_vs_eta_Pixel", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_Y_vs_Eta");
    psd0.add<std::string>("title", ";#eta;pull y");
    psd0.add<double>("ymax", 4.0);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    psd0.add<double>("ymin", -4.0);
    desc.add<edm::ParameterSetDescription>("Pull_Y_vs_eta_Pixel", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Number_RecHits_matched_PrimarySimTrack");
    psd0.add<std::string>("title", "Number of RecHits matched to primary SimTrack;;");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 10000.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("nRecHits_Pixel_primary", psd0);
  }

  //strip sensors
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X_Strip");
    psd0.add<std::string>("title", "#Delta X " + striptag + ";Cluster resolution X coordinate [#mum]");
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmin", -250);
    psd0.add<double>("xmax", 250);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Delta_X_Strip", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_Strip");
    psd0.add<std::string>("title", "#Delta Y " + striptag + ";Cluster resolution Y coordinate [cm]");
    psd0.add<double>("xmin", -5.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 5.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Delta_Y_Strip", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X_Strip_Primary");
    psd0.add<std::string>("title", "#Delta X " + striptag + ";Cluster resolution X coordinate [#mum]");
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmin", -250);
    psd0.add<double>("xmax", 250);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Delta_X_Strip_Primary", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_Strip_Primary");
    psd0.add<std::string>("title", "#Delta Y " + striptag + ";Cluster resolution Y coordinate [cm]");
    psd0.add<double>("xmin", -5.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 5.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Delta_Y_Strip_Primary", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X_vs_Eta_Strip");
    psd0.add<std::string>("title", ";#eta;#Delta x [#mum]");
    psd0.add<double>("ymin", -250.0);
    psd0.add<double>("ymax", 250.0);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    desc.add<edm::ParameterSetDescription>("Delta_X_vs_eta_Strip", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_vs_Eta_Strip");
    psd0.add<std::string>("title", ";#eta;#Delta y [cm]");
    psd0.add<double>("ymin", -5.0);
    psd0.add<double>("ymax", 5.0);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    desc.add<edm::ParameterSetDescription>("Delta_Y_vs_eta_Strip", psd0);
  }
  //pulls strips
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_X_Strip");
    psd0.add<std::string>("title", ";pull x;");
    psd0.add<double>("xmin", -4.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Pull_X_Strip", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_Y_Strip");
    psd0.add<std::string>("title", ";pull y;");
    psd0.add<double>("xmin", -4.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Pull_Y_Strip", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_X_Strip_Primary");
    psd0.add<std::string>("title", ";pull x;");
    psd0.add<double>("xmin", -4.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Pull_X_Strip_Primary", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_Y_Strip_Primary");
    psd0.add<std::string>("title", ";pull y;");
    psd0.add<double>("xmin", -4.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Pull_Y_Strip_Primary", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_X_vs_Eta_Strip");
    psd0.add<std::string>("title", ";#eta;pull x");
    psd0.add<double>("ymax", 4.0);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    psd0.add<double>("ymin", -4.0);
    desc.add<edm::ParameterSetDescription>("Pull_X_vs_eta_Strip", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_Y_vs_Eta_Strip");
    psd0.add<std::string>("title", ";#eta;pull y");
    psd0.add<double>("ymax", 4.0);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    psd0.add<double>("ymin", -4.0);
    desc.add<edm::ParameterSetDescription>("Pull_Y_vs_eta_Strip", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Number_RecHits_matched_PrimarySimTrack");
    psd0.add<std::string>("title", "Number of RecHits matched to primary SimTrack;;");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 10000.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("nRecHits_Strip_primary", psd0);
  }
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
  desc.add<edm::InputTag>("OuterTrackerDigiSimLinkSource", edm::InputTag("simSiPixelDigis", "Tracker"));
  desc.add<edm::InputTag>("OuterTrackerDigiSource", edm::InputTag("mix", "Tracker"));
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
