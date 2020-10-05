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
#include "Validation/SiTrackerPhase2V/plugins/Phase2ITValidateRecHit.h"
#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2ValidationUtil.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
//--- for SimHit association
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
Phase2ITValidateRecHit::Phase2ITValidateRecHit(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      trackerHitAssociatorConfig_(iConfig, consumesCollector()),
      simtrackminpt_(iConfig.getParameter<double>("SimTrackMinPt")),
      tokenRecHitsIT_(consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("rechitsSrc"))),
      simTracksToken_(consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("simTracksSrc"))) {
  edm::LogInfo("Phase2ITValidateRecHit") << ">>> Construct Phase2ITValidateRecHit ";
  for (const auto& itag : config_.getParameter<std::vector<edm::InputTag>>("PSimHitSource"))
    simHitTokens_.push_back(consumes<edm::PSimHitContainer>(itag));
}
//
Phase2ITValidateRecHit::~Phase2ITValidateRecHit() {
  edm::LogInfo("Phase2ITValidateRecHit") << ">>> Destroy Phase2ITValidateRecHit ";
}
void Phase2ITValidateRecHit::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the geometry
  edm::ESHandle<TrackerGeometry> geomHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(geomHandle);
  const TrackerGeometry* tkGeom = &(*geomHandle);

  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* tTopo = tTopoHandle.product();

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
  for (edm::SimTrackContainer::const_iterator simTrackIt(simTracks->begin()); simTrackIt != simTracks->end();
       ++simTrackIt) {
    if (simTrackIt->momentum().pt() > simtrackminpt_) {
      selectedSimTrackMap.insert(std::make_pair(simTrackIt->trackId(), *simTrackIt));
    }
  }
  TrackerHitAssociator associateRecHit(iEvent, trackerHitAssociatorConfig_);
  fillITHistos(iEvent, tTopo, tkGeom, associateRecHit, simHits, selectedSimTrackMap);
}

void Phase2ITValidateRecHit::fillITHistos(const edm::Event& iEvent,
                                          const TrackerTopology* tTopo,
                                          const TrackerGeometry* tkGeom,
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
  SiPixelRecHitCollection::const_iterator DSViter;
  for (DSViter = rechits->begin(); DSViter != rechits->end(); ++DSViter) {
    // Get the detector unit's id
    unsigned int rawid(DSViter->detId());
    DetId detId(rawid);
    // Get the geomdet
    const GeomDetUnit* geomDetunit(tkGeom->idToDetUnit(detId));
    if (!geomDetunit)
      continue;
    // determine the detector we are in
    std::string key = Phase2TkUtil::getITHistoId(detId.rawId(), tTopo);
    if (nrechitLayerMap_primary.find(key) == nrechitLayerMap_primary.end()) {
      nrechitLayerMap_primary.insert(std::make_pair(key, DSViter->size()));
    } else {
      nrechitLayerMap_primary[key] += DSViter->size();
    }

    edmNew::DetSet<SiPixelRecHit>::const_iterator rechitIt;
    //loop over rechits for a single detId
    for (rechitIt = DSViter->begin(); rechitIt != DSViter->end(); ++rechitIt) {
      //GetSimHits
      const std::vector<SimHitIdpr>& matchedId = associateRecHit.associateHitId(*rechitIt);
      const PSimHit* simhitClosest = nullptr;
      float minx = 10000;
      LocalPoint lp = rechitIt->localPosition();
      for (unsigned int si = 0; si < simHits.size(); ++si) {
        for (edm::PSimHitContainer::const_iterator simhitIt = simHits.at(si)->begin();
             simhitIt != simHits.at(si)->end();
             ++simhitIt) {
          if (detId.rawId() != simhitIt->detUnitId())
            continue;
          for (auto& mId : matchedId) {
            if (simhitIt->trackId() == mId.first) {
              if (!simhitClosest || fabs(simhitIt->localPosition().x() - lp.x()) < minx) {
                minx = fabs(simhitIt->localPosition().x() - lp.x());
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
        isPrimary = Phase2TkUtil::isPrimary(simTrackIt->second, simhitClosest);
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
  for (auto& lme : nrechitLayerMap_primary) {
    layerMEs_[lme.first].numberRecHitsprimary->Fill(nrechitLayerMap_primary[lme.first]);
  }
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
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  if (theTkDigiGeomWatcher.check(iSetup)) {
    edm::ESHandle<TrackerGeometry> geom_handle;
    iSetup.get<TrackerDigiGeometryRecord>().get(geomType_, geom_handle);
    const TrackerGeometry* tGeom = geom_handle.product();
    for (auto const& det_u : tGeom->detUnits()) {
      //Always check TrackerNumberingBuilder before changing this part
      if (!(det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXB ||
            det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC))
        continue;
      unsigned int detId_raw = det_u->geographicalId().rawId();
      bookLayerHistos(ibooker, detId_raw, tTopo, top_folder);
    }
  }
}
//
void Phase2ITValidateRecHit::bookLayerHistos(DQMStore::IBooker& ibooker,
                                             unsigned int det_id,
                                             const TrackerTopology* tTopo,
                                             std::string& subdir) {
  ibooker.cd();
  std::string key = Phase2TkUtil::getITHistoId(det_id, tTopo);
  if (key.empty())
    return;
  if (layerMEs_.find(key) == layerMEs_.end()) {
    ibooker.cd();
    RecHitME local_histos;
    std::ostringstream histoName;
    ibooker.setCurrentFolder(subdir + "/" + key);
    std::cout << "Setting subfolder>>>" << subdir << "\t" << key << std::endl;
    edm::LogInfo("Phase2ITValidateRecHit") << " Booking Histograms in : " << key;

    histoName.str("");
    histoName << "Delta_X";
    local_histos.deltaX =
        Phase2TkUtil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaX"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Delta_Y";
    local_histos.deltaY =
        Phase2TkUtil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("DeltaX"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Pull_X";
    local_histos.pullX =
        Phase2TkUtil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("PullX"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Pull_Y";
    local_histos.pullY =
        Phase2TkUtil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("PullY"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Delta_X_vs_Eta";
    local_histos.deltaX_eta = Phase2TkUtil::bookProfile1DFromPSet(
        config_.getParameter<edm::ParameterSet>("DeltaX_eta"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Delta_Y_vs_Eta";
    local_histos.deltaY_eta = Phase2TkUtil::bookProfile1DFromPSet(
        config_.getParameter<edm::ParameterSet>("DeltaX_eta"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Pull_X_vs_Eta";
    local_histos.pullX_eta = Phase2TkUtil::bookProfile1DFromPSet(
        config_.getParameter<edm::ParameterSet>("PullX_eta"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Pull_Y_vs_Eta";
    local_histos.pullY_eta = Phase2TkUtil::bookProfile1DFromPSet(
        config_.getParameter<edm::ParameterSet>("PullY_eta"), histoName.str(), ibooker);
    ibooker.setCurrentFolder(subdir + "/" + key + "/PrimarySimHits");
    //all histos for Primary particles
    histoName.str("");
    histoName << "Number_RecHits_matched_PrimarySimTrack";
    local_histos.numberRecHitsprimary = Phase2TkUtil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("nRecHits_primary"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Delta_X_SimHitPrimary";
    local_histos.deltaX_primary = Phase2TkUtil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("DeltaX_primary"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Delta_Y_SimHitPrimary";
    local_histos.deltaY_primary = Phase2TkUtil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("DeltaY_primary"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Pull_X_SimHitPrimary";
    local_histos.pullX_primary = Phase2TkUtil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("PullX_primary"), histoName.str(), ibooker);
    histoName.str("");
    histoName << "Pull_Y_SimHitPrimary";
    local_histos.pullY_primary = Phase2TkUtil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("PullY_primary"), histoName.str(), ibooker);
    layerMEs_.insert(std::make_pair(key, local_histos));
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2ITValidateRecHit);
