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
#include "Validation/SiTrackerPhase2V/interface/Phase2ValidationUtil.h"
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
//
// constructors
//
Phase2ITValidateRecHit::Phase2ITValidateRecHit(const edm::ParameterSet& iConfig):
  config_(iConfig),
  trackerHitAssociatorConfig_(iConfig, consumesCollector()),
  simtrackminpt_(iConfig.getParameter<double>("SimTrackMinPt")),
  tokenRecHitsIT_(consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("rechitsSrc"))),
  simTracksToken_(consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("simTracksSrc")))
{      
  edm::LogInfo("Phase2ITValidateRecHit") << ">>> Construct Phase2ITValidateRecHit ";
  for (const auto& itag :  config_.getParameter<std::vector<edm::InputTag> >("PSimHitSource"))
    simHitTokens_.push_back(consumes<edm::PSimHitContainer>(itag));
}

//
// destructor
//
Phase2ITValidateRecHit::~Phase2ITValidateRecHit() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  edm::LogInfo("Phase2ITValidateRecHit") << ">>> Destroy Phase2ITValidateRecHit ";
}
//
// -- DQM Begin Run
//
// -- Analyze
//
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
  if(!rechits.isValid())  return;
  std::map<std::string, unsigned int>  nrechitLayerMap;
  std::map<std::string, unsigned int>  nrechitLayerMap_primary;
  unsigned long int nTotrechitsinevt = 0;
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
    nTotrechitsinevt += DSViter->size();
    if(nrechitLayerMap.find(key) == nrechitLayerMap.end()) {
	nrechitLayerMap.insert(std::make_pair(key, DSViter->size())); 
	nrechitLayerMap_primary.insert(std::make_pair(key, DSViter->size())); 
    } else {
      nrechitLayerMap[key] += DSViter->size();
      nrechitLayerMap_primary[key] += DSViter->size();
    }
    
    edmNew::DetSet<SiPixelRecHit>::const_iterator rechitIt;
    //loop over rechits for a single detId
    for(rechitIt = DSViter->begin(); rechitIt != DSViter->end(); ++rechitIt) {
      LocalPoint lp = rechitIt->localPosition();
      Global3DPoint globalPos = geomDetunit->surface().toGlobal(lp);
      //in mm
      double gx = globalPos.x()*10.;
      double gy = globalPos.y()*10.;
      double gz = globalPos.z()*10.;
      double gr = globalPos.perp()*10.;
      //Fill global positions
      globalXY_->Fill(gx, gy);
      globalRZ_->Fill(gz, gr);
      //layer wise histo
      layerMEs_[key].clusterSize->Fill(rechitIt->cluster()->size());
      layerMEs_[key].globalPosXY->Fill(gx, gy); 
      layerMEs_[key].localPosXY->Fill(lp.x(), lp.y()); 
   
      //GetSimHits
      const std::vector<SimHitIdpr>& matchedId = associateRecHit.associateHitId(*rechitIt);
      //std::cout << "Nmatched SimHits = "  << matchedId.size() << ",";
      const PSimHit* simhitClosest = 0;
      float minx = 10000;
      for (unsigned int si = 0; si < simHits.size(); ++si) {
	for(edm::PSimHitContainer::const_iterator simhitIt = simHits.at(si)->begin();
	    simhitIt != simHits.at(si)->end(); ++simhitIt) {
	  if(detId.rawId() != simhitIt->detUnitId())   continue;
	  for(auto& mId : matchedId) {
	    if(simhitIt->trackId() == mId.first) {
	      if (!simhitClosest || fabs(simhitIt->localPosition().x() - lp.x()) < minx) {
		minx = fabs(simhitIt->localPosition().x() - lp.x());
		simhitClosest = &*simhitIt;
	      }
	    }
	  }
	}//end loop over PSimhitcontainers
      }//end loop over simHits
      if(!simhitClosest)   continue;
      auto simTrackIt(selectedSimTrackMap.find(simhitClosest->trackId()));
      bool isPrimary = false;
      //check if simhit is primary
      if(simTrackIt != selectedSimTrackMap.end()) isPrimary = Phase2TkUtil::isPrimary(simTrackIt->second, simhitClosest);
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
      float eta = globalPos.eta();
      layerMEs_[key].deltaX->Fill(dx);
      layerMEs_[key].deltaY->Fill(dy);
      layerMEs_[key].pullX->Fill(pullx);
      layerMEs_[key].pullY->Fill(pully);
      layerMEs_[key].deltaX_eta->Fill(eta,dx);
      layerMEs_[key].deltaY_eta->Fill(eta,dy);
      layerMEs_[key].pullX_eta->Fill(eta,pullx);
      layerMEs_[key].pullY_eta->Fill(eta,pully);
      if(isPrimary) {
	  layerMEs_[key].deltaX_primary->Fill(dx);
	  layerMEs_[key].deltaY_primary->Fill(dy);
	  layerMEs_[key].pullX_primary->Fill(pullx);
	  layerMEs_[key].pullY_primary->Fill(pully);
      } else 
	nrechitLayerMap_primary[key]--;
    }//end loop over rechits of a detId
  } //End loop over DetSetVector
 
  //fill nRecHits per event
  numberRecHits_->Fill(nTotrechitsinevt);
  //fill nRecHit counter per layer
  for(auto& lme : nrechitLayerMap) {
    layerMEs_[lme.first].numberRecHits->Fill(lme.second);
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
  //std::stringstream folder_name;
  
  ibooker.cd();
  //edm::LogInfo("Phase2ITValidateRecHit") << " Booking Histograms in : " << folder_name.str();
  std::cout << "Booking Histograms in : " << top_folder << std::endl;
  std::string dir = top_folder;
  ibooker.setCurrentFolder(dir);

  std::stringstream HistoName;
  //Global histos for OT
  HistoName.str("");
  HistoName << "NumberRecHits";
  numberRecHits_ = ibooker.book1D(HistoName.str(), HistoName.str(), 50, 0., 0.);
  HistoName.str("");
  HistoName << "Global_Position_XY_IT";
  globalXY_   = ibooker.book2D(HistoName.str(), HistoName.str(), 1250, -1250., 1250., 1250, -1250., 1250.);
  HistoName.str("");
  HistoName << "Global_Position_RZ_IT";
  globalRZ_   = ibooker.book2D(HistoName.str(), HistoName.str(), 1500, -3000., 3000., 1250., 0., 1250);

  //Now book layer wise histos

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
      if(det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXB
	 || det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC)
	continue;
      unsigned int detId_raw = det_u->geographicalId().rawId();
      std::cout << "Detid:" << detId_raw
		<<"\tsubdet=" << det_u->subDetector()
		<< "\t key=" << Phase2TkUtil::getITHistoId(detId_raw, tTopo)
		<< std::endl;
      bookLayerHistos(ibooker, detId_raw, tTopo, dir);
    }
  }

}

//
// -- Book Layer Histograms
//
void Phase2ITValidateRecHit::bookLayerHistos(DQMStore::IBooker& ibooker,
					     unsigned int det_id,
					     const TrackerTopology* tTopo,
					     std::string& subdir) {
  
  int layer, side;
  int idisc {0};
  layer = tTopo->getITPixelLayerNumber(det_id);
  if (layer < 0)
    return;
  std::string key = Phase2TkUtil::getITHistoId(det_id, tTopo);

  //std::map<std::string, RecHitME>::iterator pos = layerMEs_.find(key);
  if (layerMEs_.find(key) == layerMEs_.end()) {
    if (layer > 100) {
      side = layer / 100;
      idisc = layer - side * 100;
      idisc = (idisc < 3) ? 12 : 345;
    }
    
    bool forDisc12UptoRing10 = (idisc == 12 && tTopo->tidRing(det_id) <= 10) ? true : false;
    bool forDisc345UptoRing7 = (idisc == 345 && tTopo->tidRing(det_id) <= 7) ? true : false;
    //bool forS = (pixelFlag_) ? false : true;
    //this handles the PSP 
    bool forP = (layer < 4 || (layer > 6 && (forDisc12UptoRing10 || forDisc345UptoRing7))) ? true : false;
    
    ibooker.cd();
    RecHitME local_histos;
    std::ostringstream histoName;
    ibooker.setCurrentFolder(subdir+"/"+key);
    std::cout<< "Setting subfolder>>>" << subdir << "\t" << key << std::endl;
    edm::LogInfo("Phase2ITValidateRecHit") << " Booking Histograms in : " << key;

    histoName.str("");
    histoName << "Number_RecHits";
    local_histos.numberRecHits = ibooker.book1D(histoName.str(), histoName.str(), 50, 0., 0.);
    
    histoName.str("");
    histoName << "Cluster_Size";
    local_histos.clusterSize = ibooker.book1D(histoName.str(), histoName.str(), 21, -0.5, 20.5);
    
    histoName.str("");
    histoName << "Globalosition_XY"; 
    local_histos.globalPosXY = ibooker.book2D(histoName.str(), histoName.str(), 1250, -1250., 1250., 1250, -1250., 1250.);
    
    histoName.str("");
    histoName << "Local_Position_XY_P" ;
    local_histos.localPosXY = ibooker.book2D(histoName.str(), histoName.str(), 500, 0., 0., 500, 0., 0.);
    
    histoName.str("");
    histoName << "Delta_X";
    local_histos.deltaX = ibooker.book1D(histoName.str(), histoName.str(), 100, -0.02, 0.02);
    
    histoName.str("");
    histoName << "Delta_Y";
    local_histos.deltaY = ibooker.book1D(histoName.str(), histoName.str(), 100, -0.02, 0.02);
    
    histoName.str("");
    histoName << "Pull_X";
    local_histos.pullX = ibooker.book1D(histoName.str(), histoName.str(), 100, -4., 4.);
    
    histoName.str("");
    histoName << "Pull_Y";
    local_histos.pullY = ibooker.book1D(histoName.str(), histoName.str(), 100, -4., 4.);
      
    histoName.str("");
    histoName << "Delta_X_vs_Eta";
    local_histos.deltaX_eta = ibooker.bookProfile(histoName.str(), histoName.str(), 82, -4.1, 4.1, 100, -0.02, 0.02);
    
    histoName.str("");
    histoName << "Delta_Y_vs_Eta";
    local_histos.deltaY_eta = ibooker.bookProfile(histoName.str(), histoName.str(), 82, -4.1, 4.1, 100, -0.02, 0.02);
    
    histoName.str("");
    histoName << "Pull_X_vs_Eta";
    local_histos.pullX_eta = ibooker.bookProfile(histoName.str(), histoName.str(), 82, -4.1, 4.1, 100, -4., 4.);
    
    histoName.str("");
    histoName << "Pull_Y_vs_Eta";
    local_histos.pullY_eta = ibooker.bookProfile(histoName.str(), histoName.str(), 82, -4.1, 4.1, 100, -4., 4.);
    
    ibooker.setCurrentFolder(subdir+"/"+key+"/PrimarySimHits");
    //all histos for Primary particles
    histoName.str("");
    histoName << "Number_RecHits_matched_PrimarySimTrack_P";
    local_histos.numberRecHitsprimary = ibooker.book1D(histoName.str(), histoName.str(), 50, 0., 0.);
    
    histoName.str("");
    histoName << "Delta_X_SimHitPrimary";
    local_histos.deltaX_primary = ibooker.book1D(histoName.str(), histoName.str(), 100, -0.02, 0.02);
    
    histoName.str("");
    histoName << "Delta_Y_SimHitPrimary";
    local_histos.deltaY_primary = ibooker.book1D(histoName.str(), histoName.str(), 100, -0.02, 0.02);
    
    histoName.str("");
    histoName << "Pull_X_SimHitPrimary";
    local_histos.pullX_primary = ibooker.book1D(histoName.str(), histoName.str(), 100, -3., 3.);
    
    histoName.str("");
    histoName << "Pull_Y_SimHitPrimary";
    local_histos.pullY_primary = ibooker.book1D(histoName.str(), histoName.str(), 100, -3., 3.);
    
    layerMEs_.insert(std::make_pair(key, local_histos));
  }
}
  
//define this as a plug-in
DEFINE_FWK_MODULE(Phase2ITValidateRecHit);
 
