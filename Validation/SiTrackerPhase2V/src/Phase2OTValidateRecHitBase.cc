// Package:    Phase2OTValidateRecHitBase
// Class:      Phase2OTValidateRecHitBase
//
/**\class Phase2OTValidateRecHitBase Phase2OTValidateRecHitBase.cc 
 Description:  Standalone  Plugin for Phase2 RecHit validation
*/
//
// Author: Suvankar Roy Chowdhury
// Date: May 2021
//
// system include files
#include "Validation/SiTrackerPhase2V/interface/Phase2OTValidateRecHitBase.h"
#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2ValidationUtil.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

//
// constructors
//
Phase2OTValidateRecHitBase::Phase2OTValidateRecHitBase(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  edm::LogInfo("Phase2OTValidateRecHitBase") << ">>> Construct Phase2OTValidateRecHitBase ";
}

//
// destructor
//
Phase2OTValidateRecHitBase::~Phase2OTValidateRecHitBase() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  edm::LogInfo("Phase2OTValidateRecHitBase") << ">>> Destroy Phase2OTValidateRecHitBase ";
}
//
// -- DQM Begin Run
void Phase2OTValidateRecHitBase::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
}

void Phase2OTValidateRecHitBase::fillOTRecHitHistos(const PSimHit* simhitClosest,
                                                    const Phase2TrackerRecHit1D* rechit,
                                                    const std::map<unsigned int, SimTrack>& selectedSimTrackMap,
                                                    std::map<std::string, unsigned int>& nrechitLayerMapP_primary,
                                                    std::map<std::string, unsigned int>& nrechitLayerMapS_primary) {
  auto detId = rechit->geographicalId();
  // Get the geomdet
  const GeomDetUnit* geomDetunit(tkGeom_->idToDetUnit(detId));
  if (!geomDetunit)
    return;
  // determine the detector we are in
  TrackerGeometry::ModuleType mType = tkGeom_->getDetectorType(detId);
  std::string key = phase2tkutil::getOTHistoId(detId.rawId(), tTopo_);

  LocalPoint lp = rechit->localPosition();
  auto simTrackIt(selectedSimTrackMap.find(simhitClosest->trackId()));
  bool isPrimary = false;
  //check if simhit is primary
  if (simTrackIt != selectedSimTrackMap.end())
    isPrimary = phase2tkutil::isPrimary(simTrackIt->second, simhitClosest);
  Local3DPoint simlp(simhitClosest->localPosition());
  const LocalError& lperr = rechit->localPositionError();
  double dx = lp.x() - simlp.x();
  double dy = lp.y() - simlp.y();
  double pullx = 999.;
  double pully = 999.;
  if (lperr.xx())
    pullx = (dx) / std::sqrt(lperr.xx());
  if (lperr.yy())
    pully = (dx) / std::sqrt(lperr.yy());
  float eta = geomDetunit->surface().toGlobal(lp).eta();
  float phi = geomDetunit->surface().toGlobal(lp).phi();

  //scale for plotting
  dx *= phase2tkutil::cmtomicron;  //this is always the case
  if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
    dy *= phase2tkutil::cmtomicron;  //only for PSP sensors

    layerMEs_[key].deltaX_P->Fill(dx);
    layerMEs_[key].deltaY_P->Fill(dy);

    layerMEs_[key].pullX_P->Fill(pullx);
    layerMEs_[key].pullY_P->Fill(pully);

    layerMEs_[key].deltaX_eta_P->Fill(std::abs(eta), dx);
    layerMEs_[key].deltaY_eta_P->Fill(std::abs(eta), dy);
    layerMEs_[key].deltaX_phi_P->Fill(phi, dx);
    layerMEs_[key].deltaY_phi_P->Fill(phi, dy);

    layerMEs_[key].pullX_eta_P->Fill(eta, pullx);
    layerMEs_[key].pullY_eta_P->Fill(eta, pully);

    if (isPrimary) {
      layerMEs_[key].deltaX_primary_P->Fill(dx);
      layerMEs_[key].deltaY_primary_P->Fill(dy);
      layerMEs_[key].pullX_primary_P->Fill(pullx);
      layerMEs_[key].pullY_primary_P->Fill(pully);
    } else
      nrechitLayerMapP_primary[key]--;
  } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
    layerMEs_[key].deltaX_S->Fill(dx);
    layerMEs_[key].deltaY_S->Fill(dy);

    layerMEs_[key].pullX_S->Fill(pullx);
    layerMEs_[key].pullY_S->Fill(pully);

    layerMEs_[key].deltaX_eta_S->Fill(std::abs(eta), dx);
    layerMEs_[key].deltaY_eta_S->Fill(std::abs(eta), dy);
    layerMEs_[key].deltaX_phi_S->Fill(phi, dx);
    layerMEs_[key].deltaY_phi_S->Fill(phi, dy);

    layerMEs_[key].pullX_eta_S->Fill(eta, pullx);
    layerMEs_[key].pullY_eta_S->Fill(eta, pully);

    if (isPrimary) {
      layerMEs_[key].deltaX_primary_S->Fill(dx);
      layerMEs_[key].deltaY_primary_S->Fill(dy);
      layerMEs_[key].pullX_primary_S->Fill(pullx);
      layerMEs_[key].pullY_primary_S->Fill(pully);
    } else
      nrechitLayerMapS_primary[key]--;
  }
}
//
// -- Book Histograms
//
void Phase2OTValidateRecHitBase::bookHistograms(DQMStore::IBooker& ibooker,
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
void Phase2OTValidateRecHitBase::bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir) {
  std::string key = phase2tkutil::getOTHistoId(det_id, tTopo_);
  if (layerMEs_.find(key) == layerMEs_.end()) {
    ibooker.cd();
    RecHitME local_histos;
    ibooker.setCurrentFolder(subdir + "/" + key);
    edm::LogInfo("Phase2OTValidateRecHitBase") << " Booking Histograms in : " << key;

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
          phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_vs_eta_Pixel"), ibooker);
      local_histos.deltaY_eta_P =
          phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_Y_vs_eta_Pixel"), ibooker);

      local_histos.deltaX_phi_P =
          phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_vs_phi_Pixel"), ibooker);
      local_histos.deltaY_phi_P =
          phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_Y_vs_phi_Pixel"), ibooker);

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
        phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_vs_eta_Strip"), ibooker);
    local_histos.deltaY_eta_S =
        phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_Y_vs_eta_Strip"), ibooker);

    local_histos.deltaX_phi_S =
        phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_vs_phi_Strip"), ibooker);
    local_histos.deltaY_phi_S =
        phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_Y_vs_phi_Strip"), ibooker);

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

void Phase2OTValidateRecHitBase::fillPSetDescription(edm::ParameterSetDescription& desc) {
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
    psd0.add<std::string>("title", ";|#eta|;#Delta x [#mum]");
    psd0.add<int>("NyBins", 250);
    psd0.add<double>("ymin", -250.0);
    psd0.add<double>("ymax", 250.0);
    psd0.add<int>("NxBins", 41);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", 0.);
    desc.add<edm::ParameterSetDescription>("Delta_X_vs_eta_Pixel", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_vs_Eta_Pixel");
    psd0.add<std::string>("title", ";|#eta|;#Delta y [#mum]");
    psd0.add<int>("NyBins", 300);
    psd0.add<double>("ymin", -1500.0);
    psd0.add<double>("ymax", 1500.0);
    psd0.add<int>("NxBins", 41);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", 0.);
    desc.add<edm::ParameterSetDescription>("Delta_Y_vs_eta_Pixel", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X_vs_Phi_Pixel");
    psd0.add<std::string>("title", ";#phi;#Delta x [#mum]");
    psd0.add<int>("NyBins", 250);
    psd0.add<double>("ymin", -250.0);
    psd0.add<double>("ymax", 250.0);
    psd0.add<int>("NxBins", 36);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", M_PI);
    psd0.add<double>("xmin", -M_PI);
    desc.add<edm::ParameterSetDescription>("Delta_X_vs_phi_Pixel", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_vs_Phi_Pixel");
    psd0.add<std::string>("title", ";#phi;#Delta y [#mum]");
    psd0.add<int>("NyBins", 300);
    psd0.add<double>("ymin", -1500.0);
    psd0.add<double>("ymax", 1500.0);
    psd0.add<int>("NxBins", 35);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", M_PI);
    psd0.add<double>("xmin", -M_PI);
    desc.add<edm::ParameterSetDescription>("Delta_Y_vs_phi_Pixel", psd0);
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
    psd0.add<std::string>("title", ";|#eta|;#Delta x [#mum]");
    psd0.add<int>("NyBins", 250);
    psd0.add<double>("ymin", -250.0);
    psd0.add<double>("ymax", 250.0);
    psd0.add<int>("NxBins", 41);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", 0.);
    desc.add<edm::ParameterSetDescription>("Delta_X_vs_eta_Strip", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_vs_Eta_Strip");
    psd0.add<std::string>("title", ";|#eta|;#Delta y [cm]");
    psd0.add<int>("NyBins", 100);
    psd0.add<double>("ymin", -5.0);
    psd0.add<double>("ymax", 5.0);
    psd0.add<int>("NxBins", 41);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", 0.);
    desc.add<edm::ParameterSetDescription>("Delta_Y_vs_eta_Strip", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X_vs_Phi_Strip");
    psd0.add<std::string>("title", ";#phi;#Delta x [#mum]");
    psd0.add<int>("NyBins", 250);
    psd0.add<double>("ymin", -250.0);
    psd0.add<double>("ymax", 250.0);
    psd0.add<int>("NxBins", 36);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", M_PI);
    psd0.add<double>("xmin", -M_PI);
    desc.add<edm::ParameterSetDescription>("Delta_X_vs_phi_Strip", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_vs_Phi_Strip");
    psd0.add<std::string>("title", ";#phi;#Delta y [cm]");
    psd0.add<int>("NyBins", 100);
    psd0.add<double>("ymin", -5.0);
    psd0.add<double>("ymax", 5.0);
    psd0.add<int>("NxBins", 36);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", M_PI);
    psd0.add<double>("xmin", -M_PI);
    desc.add<edm::ParameterSetDescription>("Delta_Y_vs_phi_Strip", psd0);
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
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2OTValidateRecHitBase);
