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
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//--- for SimHit association
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2ValidationUtil.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"

class Phase2ITValidateRecHit : public DQMEDAnalyzer {
public:
  explicit Phase2ITValidateRecHit(const edm::ParameterSet&);
  ~Phase2ITValidateRecHit() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void fillITHistos(const edm::Event& iEvent,
                    const TrackerHitAssociator& associateRecHit,
                    const std::vector<edm::Handle<edm::PSimHitContainer>>& simHits,
                    const std::map<unsigned int, SimTrack>& selectedSimTrackMap);

  void bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir);

  edm::ParameterSet config_;
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  const double simtrackminpt_;
  std::string geomType_;
  const edm::EDGetTokenT<SiPixelRecHitCollection> tokenRecHitsIT_;
  const edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHitTokens_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
  struct RecHitME {
    MonitorElement* deltaX = nullptr;
    MonitorElement* deltaY = nullptr;
    MonitorElement* pullX = nullptr;
    MonitorElement* pullY = nullptr;
    MonitorElement* deltaX_eta = nullptr;
    MonitorElement* deltaY_eta = nullptr;
    MonitorElement* pullX_eta = nullptr;
    MonitorElement* pullY_eta = nullptr;
    //For rechits matched to primary simhits
    MonitorElement* numberRecHitsprimary = nullptr;
    MonitorElement* pullX_primary;
    MonitorElement* pullY_primary;
    MonitorElement* deltaX_primary;
    MonitorElement* deltaY_primary;
  };
  std::map<std::string, RecHitME> layerMEs_;
};

Phase2ITValidateRecHit::Phase2ITValidateRecHit(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      trackerHitAssociatorConfig_(iConfig, consumesCollector()),
      simtrackminpt_(iConfig.getParameter<double>("SimTrackMinPt")),
      tokenRecHitsIT_(consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("rechitsSrc"))),
      simTracksToken_(consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("simTracksSrc"))),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
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
    edm::Handle<edm::PSimHitContainer> simHitHandle;
    iEvent.getByToken(itoken, simHitHandle);
    if (!simHitHandle.isValid())
      continue;
    simHits.emplace_back(simHitHandle);
  }
  // Get the SimTracks and push them in a map of id, SimTrack
  edm::Handle<edm::SimTrackContainer> simTracks;
  iEvent.getByToken(simTracksToken_, simTracks);

  std::map<unsigned int, SimTrack> selectedSimTrackMap;
  for (const auto& simTrackIt : *simTracks) {
    if (simTrackIt.momentum().pt() > simtrackminpt_) {
      selectedSimTrackMap.insert(std::make_pair(simTrackIt.trackId(), simTrackIt));
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
  edm::Handle<SiPixelRecHitCollection> rechits;
  iEvent.getByToken(tokenRecHitsIT_, rechits);
  if (!rechits.isValid())
    return;
  std::map<std::string, unsigned int> nrechitLayerMap_primary;
  // Loop over modules
  for (const auto& DSViter : *rechits) {
    // Get the detector unit's id
    unsigned int rawid(DSViter.detId());
    DetId detId(rawid);
    // Get the geomdet
    const GeomDetUnit* geomDetunit(tkGeom_->idToDetUnit(detId));
    if (!geomDetunit)
      continue;
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
                minx = abs(simhitIt.localPosition().x() - lp.x());
                simhitClosest = &simhitIt;
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
      const LocalError& lperr = rechit.localPositionError();
      double dx = lp.x() - simlp.x();
      double dy = lp.y() - simlp.y();
      double pullx = 999.;
      double pully = 999.;
      if (lperr.xx())
        pullx = (lp.x() - simlp.x()) / std::sqrt(lperr.xx());
      if (lperr.yy())
        pully = (lp.y() - simlp.y()) / std::sqrt(lperr.yy());
      float eta = geomDetunit->surface().toGlobal(lp).eta();
      layerMEs_[key].deltaX->Fill(dx);
      layerMEs_[key].deltaY->Fill(dy);
      layerMEs_[key].pullX->Fill(pullx);
      layerMEs_[key].pullY->Fill(pully);
      layerMEs_[key].deltaX_eta->Fill(eta, dx);
      layerMEs_[key].deltaY_eta->Fill(eta, dy);
      layerMEs_[key].pullX_eta->Fill(eta, pullx);
      layerMEs_[key].pullY_eta->Fill(eta, pully);
      if (isPrimary) {
        layerMEs_[key].deltaX_primary->Fill(dx);
        layerMEs_[key].deltaY_primary->Fill(dy);
        layerMEs_[key].pullX_primary->Fill(pullx);
        layerMEs_[key].pullY_primary->Fill(pully);
      } else
        nrechitLayerMap_primary[key]--;
    }  //end loop over rechits of a detId
  }    //End loop over DetSetVector

  //fill nRecHit counter per layer
  for (const auto& lme : nrechitLayerMap_primary) {
    layerMEs_[lme.first].numberRecHitsprimary->Fill(nrechitLayerMap_primary[lme.first]);
  }
}

void Phase2ITValidateRecHit::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  edm::ESHandle<TrackerGeometry> geomHandle = iSetup.getHandle(geomToken_);
  tkGeom_ = &(*geomHandle);
  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(topoToken_);
  tTopo_ = tTopoHandle.product();
}
//
// -- Book Histograms
//
void Phase2ITValidateRecHit::bookHistograms(DQMStore::IBooker& ibooker,
                                            edm::Run const& iRun,
                                            edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  edm::LogInfo("Phase2ITValidateRecHit") << " Booking Histograms in : " << top_folder;
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(iSetup)) {
    for (auto const& det_u : tkGeom_->detUnits()) {
      //Always check TrackerNumberingBuilder before changing this part
      if (!(det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXB ||
            det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC))
        continue;
      unsigned int detId_raw = det_u->geographicalId().rawId();
      bookLayerHistos(ibooker, detId_raw, top_folder);
    }
  }
}
//
void Phase2ITValidateRecHit::bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir) {
  ibooker.cd();
  std::string key = phase2tkutil::getITHistoId(det_id, tTopo_);
  if (key.empty())
    return;
  if (layerMEs_.find(key) == layerMEs_.end()) {
    ibooker.cd();
    RecHitME local_histos;
    std::ostringstream histoName;
    ibooker.setCurrentFolder(subdir + "/" + key);
    edm::LogInfo("Phase2ITValidateRecHit") << " Booking Histograms in : " << (subdir + "/" + key);
    histoName.str("");
    histoName << "Delta_X";
    local_histos.deltaX =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaX"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Delta_Y";
    local_histos.deltaY =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaY"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Pull_X";
    local_histos.pullX =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("PullX"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Pull_Y";
    local_histos.pullY =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("PullY"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Delta_X_vs_Eta";
    local_histos.deltaX_eta = phase2tkutil::bookProfile1DFromPSet(
        config_.getParameter<edm::ParameterSet>("DeltaX_eta"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Delta_Y_vs_Eta";
    local_histos.deltaY_eta = phase2tkutil::bookProfile1DFromPSet(
        config_.getParameter<edm::ParameterSet>("DeltaX_eta"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Pull_X_vs_Eta";
    local_histos.pullX_eta = phase2tkutil::bookProfile1DFromPSet(
        config_.getParameter<edm::ParameterSet>("PullX_eta"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Pull_Y_vs_Eta";
    local_histos.pullY_eta = phase2tkutil::bookProfile1DFromPSet(
        config_.getParameter<edm::ParameterSet>("PullY_eta"), histoName.str(), ibooker);
    ibooker.setCurrentFolder(subdir + "/" + key + "/PrimarySimHits");
    //all histos for Primary particles
    histoName.str("");
    histoName << "Number_RecHits_matched_PrimarySimTrack";
    local_histos.numberRecHitsprimary = phase2tkutil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("nRecHits_primary"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Delta_X_SimHitPrimary";
    local_histos.deltaX_primary = phase2tkutil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("DeltaX_primary"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Delta_Y_SimHitPrimary";
    local_histos.deltaY_primary = phase2tkutil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("DeltaY_primary"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Pull_X_SimHitPrimary";
    local_histos.pullX_primary = phase2tkutil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("PullX_primary"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Pull_Y_SimHitPrimary";
    local_histos.pullY_primary = phase2tkutil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("PullY_primary"), histoName.str(), ibooker);
    layerMEs_.emplace(key, local_histos);
  }
}
void Phase2ITValidateRecHit::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // rechitValidIT
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X");
    psd0.add<std::string>("title", "Delta_X;RecHit resolution X dimension");
    psd0.add<double>("xmin", -0.2);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 0.2);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("DeltaX", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y");
    psd0.add<std::string>("title", "Delta_Y;RecHit resolution Y dimension;");
    psd0.add<double>("xmin", -0.2);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 0.2);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("DeltaY", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_X");
    psd0.add<std::string>("title", "Pull_X;pull x;");
    psd0.add<double>("xmin", -4.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("PullX", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_Y");
    psd0.add<std::string>("title", "Pull_Y;pull y;");
    psd0.add<double>("xmin", -4.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("PullY", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X_vs_Eta");
    psd0.add<std::string>("title", "Delta_X_vs_Eta;#eta;#Delta x");
    psd0.add<double>("ymax", 0.02);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    psd0.add<double>("ymin", -0.02);
    desc.add<edm::ParameterSetDescription>("DeltaX_eta", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_vs_Eta");
    psd0.add<std::string>("title", "Delta_Y_vs_Eta;#eta;#Delta y");
    psd0.add<double>("ymax", 0.02);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    psd0.add<double>("ymin", -0.02);
    desc.add<edm::ParameterSetDescription>("DeltaY_eta", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_X_vs_Eta");
    psd0.add<std::string>("title", "Pull_X_vs_Eta;#eta;pull x");
    psd0.add<double>("ymax", 4.0);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    psd0.add<double>("ymin", -4.0);
    desc.add<edm::ParameterSetDescription>("PullX_eta", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_Y_vs_Eta");
    psd0.add<std::string>("title", "Pull_Y_vs_Eta;#eta;pull y");
    psd0.add<double>("ymax", 4.0);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    psd0.add<double>("ymin", -4.0);
    desc.add<edm::ParameterSetDescription>("PullY_eta", psd0);
  }
  //simhits primary
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Number_RecHits_matched_PrimarySimTrack");
    psd0.add<std::string>("title", "Number of RecHits matched to primary SimTrack;;");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 0.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("nRecHits_primary", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X_SimHitPrimary");
    psd0.add<std::string>("title", "Delta_X_SimHitPrimary;#delta x;");
    psd0.add<double>("xmin", -0.2);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 0.2);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("DeltaX_primary", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_SimHitPrimary");
    psd0.add<std::string>("title", "Delta_Y_SimHitPrimary;#Delta y;");
    psd0.add<double>("xmin", -0.2);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 0.2);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("DeltaY_primary", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_X_SimHitPrimary");
    psd0.add<std::string>("title", "Pull_X_SimHitPrimary;pull x;");
    psd0.add<double>("ymax", 4.0);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    psd0.add<double>("ymin", -4.0);
    desc.add<edm::ParameterSetDescription>("PullX_primary", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Pull_Y_SimHitPrimary");
    psd0.add<std::string>("title", "Pull_Y_SimHitPrimary;pull y;");
    psd0.add<double>("ymax", 4.0);
    psd0.add<int>("NxBins", 82);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    psd0.add<double>("ymin", -4.0);
    desc.add<edm::ParameterSetDescription>("PullY_primary", psd0);
  }
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
