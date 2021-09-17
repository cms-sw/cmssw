#include "FWCore/Framework/interface/ESWatcher.h"
#include "Validation/SiTrackerPhase2V/interface/Phase2ITValidateRecHitBase.h"
#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2ValidationUtil.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"
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

Phase2ITValidateRecHitBase::~Phase2ITValidateRecHitBase() = default;

Phase2ITValidateRecHitBase::Phase2ITValidateRecHitBase(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  edm::LogInfo("Phase2ITValidateRecHitBase") << ">>> Construct Phase2ITValidateRecHitBase ";
}

void Phase2ITValidateRecHitBase::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
}
//
// -- Book Histograms
//
void Phase2ITValidateRecHitBase::bookHistograms(DQMStore::IBooker& ibooker,
                                                edm::Run const& iRun,
                                                edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  edm::LogInfo("Phase2ITValidateRecHitBase") << " Booking Histograms in : " << top_folder;
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
void Phase2ITValidateRecHitBase::bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir) {
  ibooker.cd();
  std::string key = phase2tkutil::getITHistoId(det_id, tTopo_);
  if (key.empty())
    return;
  if (layerMEs_.find(key) == layerMEs_.end()) {
    ibooker.cd();
    RecHitME local_histos;
    ibooker.setCurrentFolder(subdir + "/" + key);
    edm::LogInfo("Phase2ITValidateRecHit") << " Booking Histograms in : " << (subdir + "/" + key);

    local_histos.deltaX = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaX"), ibooker);

    local_histos.deltaY = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaY"), ibooker);

    local_histos.pullX = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("PullX"), ibooker);

    local_histos.pullY = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("PullY"), ibooker);

    local_histos.deltaX_eta =
        phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaX_eta"), ibooker);

    local_histos.deltaX_phi =
        phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaX_phi"), ibooker);

    local_histos.deltaY_eta =
        phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaY_eta"), ibooker);

    local_histos.deltaY_phi =
        phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaY_phi"), ibooker);

    local_histos.deltaX_clsizex =
        phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaX_clsizex"), ibooker);

    local_histos.deltaX_clsizey =
        phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaX_clsizey"), ibooker);

    local_histos.deltaY_clsizex =
        phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaY_clsizex"), ibooker);

    local_histos.deltaY_clsizey =
        phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaY_clsizey"), ibooker);

    local_histos.deltaYvsdeltaX =
        phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaY_vs_DeltaX"), ibooker);

    local_histos.pullX_eta =
        phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("PullX_eta"), ibooker);

    local_histos.pullY_eta =
        phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("PullY_eta"), ibooker);
    ibooker.setCurrentFolder(subdir + "/" + key + "/PrimarySimHits");
    //all histos for Primary particles
    local_histos.numberRecHitsprimary =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("nRecHits_primary"), ibooker);

    local_histos.deltaX_primary =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaX_primary"), ibooker);

    local_histos.deltaY_primary =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaY_primary"), ibooker);

    local_histos.pullX_primary =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("PullX_primary"), ibooker);

    local_histos.pullY_primary =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("PullY_primary"), ibooker);

    layerMEs_.emplace(key, local_histos);
  }
}

void Phase2ITValidateRecHitBase::fillRechitHistos(const PSimHit* simhitClosest,
                                                  const SiPixelRecHit* rechit,
                                                  const std::map<unsigned int, SimTrack>& selectedSimTrackMap,
                                                  std::map<std::string, unsigned int>& nrechitLayerMap_primary) {
  auto id = rechit->geographicalId();
  std::string key = phase2tkutil::getITHistoId(id.rawId(), tTopo_);
  const GeomDetUnit* geomDetunit(tkGeom_->idToDetUnit(id));
  if (!geomDetunit)
    return;

  LocalPoint lp = rechit->localPosition();
  auto simTrackIt(selectedSimTrackMap.find(simhitClosest->trackId()));
  bool isPrimary = false;
  //check if simhit is primary
  if (simTrackIt != selectedSimTrackMap.end())
    isPrimary = phase2tkutil::isPrimary(simTrackIt->second, simhitClosest);

  Local3DPoint simlp(simhitClosest->localPosition());
  const LocalError& lperr = rechit->localPositionError();
  double dx = phase2tkutil::cmtomicron * (lp.x() - simlp.x());
  double dy = phase2tkutil::cmtomicron * (lp.y() - simlp.y());
  double pullx = 999.;
  double pully = 999.;
  if (lperr.xx())
    pullx = (lp.x() - simlp.x()) / std::sqrt(lperr.xx());
  if (lperr.yy())
    pully = (lp.y() - simlp.y()) / std::sqrt(lperr.yy());
  float eta = geomDetunit->surface().toGlobal(lp).eta();
  float phi = geomDetunit->surface().toGlobal(lp).phi();
  layerMEs_[key].deltaX->Fill(dx);
  layerMEs_[key].deltaY->Fill(dy);
  layerMEs_[key].pullX->Fill(pullx);
  layerMEs_[key].pullY->Fill(pully);

  layerMEs_[key].deltaX_eta->Fill(std::abs(eta), dx);
  layerMEs_[key].deltaY_eta->Fill(std::abs(eta), dy);
  layerMEs_[key].deltaX_phi->Fill(phi, dx);
  layerMEs_[key].deltaY_phi->Fill(phi, dy);

  layerMEs_[key].deltaX_clsizex->Fill(rechit->cluster()->sizeX(), dx);
  layerMEs_[key].deltaX_clsizey->Fill(rechit->cluster()->sizeY(), dx);
  layerMEs_[key].deltaY_clsizex->Fill(rechit->cluster()->sizeX(), dy);
  layerMEs_[key].deltaY_clsizey->Fill(rechit->cluster()->sizeY(), dy);
  layerMEs_[key].deltaYvsdeltaX->Fill(dx, dy);
  layerMEs_[key].pullX_eta->Fill(eta, pullx);
  layerMEs_[key].pullY_eta->Fill(eta, pully);
  if (isPrimary) {
    layerMEs_[key].deltaX_primary->Fill(dx);
    layerMEs_[key].deltaY_primary->Fill(dy);
    layerMEs_[key].pullX_primary->Fill(pullx);
    layerMEs_[key].pullY_primary->Fill(pully);
  } else {
    nrechitLayerMap_primary[key]--;
  }
}

void Phase2ITValidateRecHitBase::fillPSetDescription(edm::ParameterSetDescription& desc) {
  edm::ParameterSetDescription psd0;
  psd0.add<std::string>("name", "Delta_X");
  psd0.add<std::string>("title", "Delta_X;RecHit resolution X coordinate [#mum]");
  psd0.add<bool>("switch", true);
  psd0.add<double>("xmin", -100.0);
  psd0.add<double>("xmax", 100.0);
  psd0.add<int>("NxBins", 100);
  desc.add<edm::ParameterSetDescription>("DeltaX", psd0);

  edm::ParameterSetDescription psd1;
  psd1.add<std::string>("name", "Delta_Y");
  psd1.add<std::string>("title", "Delta_Y;RecHit resolution Y coordinate [#mum];");
  psd1.add<bool>("switch", true);
  psd1.add<double>("xmin", -100.0);
  psd1.add<double>("xmax", 100.0);
  psd1.add<int>("NxBins", 100);
  desc.add<edm::ParameterSetDescription>("DeltaY", psd1);

  edm::ParameterSetDescription psd2;
  psd2.add<std::string>("name", "Pull_X");
  psd2.add<std::string>("title", "Pull_X;pull x;");
  psd2.add<double>("xmin", -4.0);
  psd2.add<bool>("switch", true);
  psd2.add<double>("xmax", 4.0);
  psd2.add<int>("NxBins", 100);
  desc.add<edm::ParameterSetDescription>("PullX", psd2);

  edm::ParameterSetDescription psd3;
  psd3.add<std::string>("name", "Pull_Y");
  psd3.add<std::string>("title", "Pull_Y;pull y;");
  psd3.add<double>("xmin", -4.0);
  psd3.add<bool>("switch", true);
  psd3.add<double>("xmax", 4.0);
  psd3.add<int>("NxBins", 100);
  desc.add<edm::ParameterSetDescription>("PullY", psd3);

  edm::ParameterSetDescription psd4;
  psd4.add<std::string>("name", "Delta_X_vs_Eta");
  psd4.add<std::string>("title", "Delta_X_vs_Eta;|#eta|;#Delta x [#mum]");
  psd4.add<int>("NyBins", 100);
  psd4.add<double>("ymin", -100.0);
  psd4.add<double>("ymax", 100.0);
  psd4.add<int>("NxBins", 41);
  psd4.add<bool>("switch", true);
  psd4.add<double>("xmax", 4.1);
  psd4.add<double>("xmin", 0.);
  desc.add<edm::ParameterSetDescription>("DeltaX_eta", psd4);

  edm::ParameterSetDescription psd4_y;
  psd4_y.add<std::string>("name", "Delta_X_vs_Phi");
  ;
  psd4_y.add<std::string>("title", "Delta_X_vs_Phi;#phi;#Delta x [#mum]");
  psd4_y.add<int>("NyBins", 100);
  psd4_y.add<double>("ymin", -100.0);
  psd4_y.add<double>("ymax", 100.0);
  psd4_y.add<int>("NxBins", 36);
  psd4_y.add<bool>("switch", true);
  psd4_y.add<double>("xmax", M_PI);
  psd4_y.add<double>("xmin", -M_PI);
  desc.add<edm::ParameterSetDescription>("DeltaX_phi", psd4_y);

  edm::ParameterSetDescription psd5;
  psd5.add<std::string>("name", "Delta_Y_vs_Eta");
  psd5.add<std::string>("title", "Delta_Y_vs_Eta;|#eta|;#Delta y [#mum]");
  psd5.add<int>("NyBins", 100);
  psd5.add<double>("ymin", -100.0);
  psd5.add<double>("ymax", 100.0);
  psd5.add<int>("NxBins", 41);
  psd5.add<bool>("switch", true);
  psd5.add<double>("xmax", 4.1);
  psd5.add<double>("xmin", 0.);
  desc.add<edm::ParameterSetDescription>("DeltaY_eta", psd5);

  edm::ParameterSetDescription psd5_y;
  psd5_y.add<std::string>("name", "Delta_Y_vs_Phi");
  psd5_y.add<std::string>("title", "Delta_Y_vs_Phi;#phi;#Delta y [#mum]");
  psd5_y.add<int>("NyBins", 100);
  psd5_y.add<double>("ymin", -100.0);
  psd5_y.add<double>("ymax", 100.0);
  psd5_y.add<int>("NxBins", 36);
  psd5_y.add<bool>("switch", true);
  psd5_y.add<double>("xmax", M_PI);
  psd5_y.add<double>("xmin", -M_PI);
  desc.add<edm::ParameterSetDescription>("DeltaY_phi", psd5_y);

  edm::ParameterSetDescription psd6;
  psd6.add<std::string>("name", "Delta_X_vs_ClusterSizeX");
  psd6.add<std::string>("title", ";Cluster size X;#Delta x [#mum]");
  psd6.add<double>("ymin", -100.0);
  psd6.add<double>("ymax", 100.0);
  psd6.add<int>("NxBins", 21);
  psd6.add<bool>("switch", true);
  psd6.add<double>("xmax", 20.5);
  psd6.add<double>("xmin", -0.5);
  desc.add<edm::ParameterSetDescription>("DeltaX_clsizex", psd6);

  edm::ParameterSetDescription psd7;
  psd7.add<std::string>("name", "Delta_X_vs_ClusterSizeY");
  psd7.add<std::string>("title", ";Cluster size Y;#Delta y [#mum]");
  psd7.add<double>("ymin", -100.0);
  psd7.add<double>("ymax", 100.0);
  psd7.add<int>("NxBins", 21);
  psd7.add<bool>("switch", true);
  psd7.add<double>("xmax", 20.5);
  psd7.add<double>("xmin", -0.5);
  desc.add<edm::ParameterSetDescription>("DeltaX_clsizey", psd7);

  edm::ParameterSetDescription psd8;
  psd8.add<std::string>("name", "Delta_Y_vs_ClusterSizeX");
  psd8.add<std::string>("title", ";Cluster size Y;#Delta y [#mum]");
  psd8.add<double>("ymin", -100.0);
  psd8.add<double>("ymax", 100.0);
  psd8.add<int>("NxBins", 21);
  psd8.add<bool>("switch", true);
  psd8.add<double>("xmax", 20.5);
  psd8.add<double>("xmin", -0.5);
  desc.add<edm::ParameterSetDescription>("DeltaY_clsizex", psd8);

  edm::ParameterSetDescription psd9;
  psd9.add<std::string>("name", "Delta_Y_vs_ClusterSizeY");
  psd9.add<std::string>("title", ";Cluster size Y;#Delta y [#mum]");
  psd9.add<double>("ymin", -100.0);
  psd9.add<double>("ymax", 100.0);
  psd9.add<int>("NxBins", 21);
  psd9.add<bool>("switch", true);
  psd9.add<double>("xmax", 20.5);
  psd9.add<double>("xmin", -0.5);
  desc.add<edm::ParameterSetDescription>("DeltaY_clsizey", psd9);

  edm::ParameterSetDescription psd10;
  psd10.add<std::string>("name", "Delta_Y_vs_DeltaX");
  psd10.add<std::string>("title", ";#Delta x[#mum];#Delta y[#mum]");
  psd10.add<bool>("switch", true);
  psd10.add<double>("ymin", -100.0);
  psd10.add<double>("ymax", 100.0);
  psd10.add<int>("NyBins", 100);
  psd10.add<double>("xmax", 100.);
  psd10.add<double>("xmin", -100.);
  psd10.add<int>("NxBins", 100);
  desc.add<edm::ParameterSetDescription>("DeltaY_vs_DeltaX", psd10);

  edm::ParameterSetDescription psd11;
  psd11.add<std::string>("name", "Pull_X_vs_Eta");
  psd11.add<std::string>("title", "Pull_X_vs_Eta;#eta;pull x");
  psd11.add<double>("ymax", 4.0);
  psd11.add<int>("NxBins", 82);
  psd11.add<bool>("switch", true);
  psd11.add<double>("xmax", 4.1);
  psd11.add<double>("xmin", -4.1);
  psd11.add<double>("ymin", -4.0);
  desc.add<edm::ParameterSetDescription>("PullX_eta", psd11);

  edm::ParameterSetDescription psd12;
  psd12.add<std::string>("name", "Pull_Y_vs_Eta");
  psd12.add<std::string>("title", "Pull_Y_vs_Eta;#eta;pull y");
  psd12.add<double>("ymax", 4.0);
  psd12.add<int>("NxBins", 82);
  psd12.add<bool>("switch", true);
  psd12.add<double>("xmax", 4.1);
  psd12.add<double>("xmin", -4.1);
  psd12.add<double>("ymin", -4.0);
  desc.add<edm::ParameterSetDescription>("PullY_eta", psd12);

  //simhits primary

  edm::ParameterSetDescription psd13;
  psd13.add<std::string>("name", "Number_RecHits_matched_PrimarySimTrack");
  psd13.add<std::string>("title", "Number of RecHits matched to primary SimTrack;;");
  psd13.add<double>("xmin", 0.0);
  psd13.add<bool>("switch", true);
  psd13.add<double>("xmax", 0.0);
  psd13.add<int>("NxBins", 100);
  desc.add<edm::ParameterSetDescription>("nRecHits_primary", psd13);

  edm::ParameterSetDescription psd14;
  psd14.add<std::string>("name", "Delta_X_SimHitPrimary");
  psd14.add<std::string>("title", "Delta_X_SimHitPrimary;#delta x [#mum];");
  psd14.add<double>("xmin", -100.0);
  psd14.add<bool>("switch", true);
  psd14.add<double>("xmax", 100.0);
  psd14.add<int>("NxBins", 100);
  desc.add<edm::ParameterSetDescription>("DeltaX_primary", psd14);

  edm::ParameterSetDescription psd15;
  psd15.add<std::string>("name", "Delta_Y_SimHitPrimary");
  psd15.add<std::string>("title", "Delta_Y_SimHitPrimary;#Delta y [#mum];");
  psd15.add<double>("xmin", -100.0);
  psd15.add<bool>("switch", true);
  psd15.add<double>("xmax", 100.0);
  psd15.add<int>("NxBins", 100);
  desc.add<edm::ParameterSetDescription>("DeltaY_primary", psd15);

  edm::ParameterSetDescription psd16;
  psd16.add<std::string>("name", "Pull_X_SimHitPrimary");
  psd16.add<std::string>("title", "Pull_X_SimHitPrimary;pull x;");
  psd16.add<double>("ymax", 4.0);
  psd16.add<int>("NxBins", 82);
  psd16.add<bool>("switch", true);
  psd16.add<double>("xmax", 4.1);
  psd16.add<double>("xmin", -4.1);
  psd16.add<double>("ymin", -4.0);
  desc.add<edm::ParameterSetDescription>("PullX_primary", psd16);

  edm::ParameterSetDescription psd17;
  psd17.add<std::string>("name", "Pull_Y_SimHitPrimary");
  psd17.add<std::string>("title", "Pull_Y_SimHitPrimary;pull y;");
  psd17.add<double>("ymax", 4.0);
  psd17.add<int>("NxBins", 82);
  psd17.add<bool>("switch", true);
  psd17.add<double>("xmax", 4.1);
  psd17.add<double>("xmin", -4.1);
  psd17.add<double>("ymin", -4.0);
  desc.add<edm::ParameterSetDescription>("PullY_primary", psd17);
}
