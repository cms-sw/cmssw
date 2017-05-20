// -*- C++ -*-
//
// Package:    Phase2TrackerValidateDigi
// Class:      Phase2TrackerValidateDigi
// 
/**\class Phase2TrackerValidateDigi Phase2TrackerValidateDigi.cc 

 Description: Test pixel digis. 

*/
//
// Author: Suchandra Dutta, Suvankar Roy Chowdhury, Subir Sarkar
// Date: January 29, 2016
//
// system include files
#include <memory>
#include "SimTracker/SiPhase2Digitizer/test/Phase2TrackerValidateDigi.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerFwd.h"

// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"

// // constructors 
//
Phase2TrackerValidateDigi::Phase2TrackerValidateDigi(const edm::ParameterSet& iConfig) :
  config_(iConfig),
  pixelFlag_(config_.getParameter<bool >("PixelPlotFillingFlag")),
  geomType_(config_.getParameter<std::string>("GeometryType")),
  otDigiSrc_(config_.getParameter<edm::InputTag>("OuterTrackerDigiSource")),
  otDigiSimLinkSrc_(config_.getParameter<edm::InputTag>("OuterTrackerDigiSimLinkSource")),
  itPixelDigiSrc_(config_.getParameter<edm::InputTag>("InnerPixelDigiSource")),
  itPixelDigiSimLinkSrc_(config_.getParameter<edm::InputTag>("InnerPixelDigiSimLinkSource")),
  pSimHitSrc_(config_.getParameter<std::vector<edm::InputTag> >("PSimHitSource")),
  simTrackSrc_(config_.getParameter<edm::InputTag>("SimTrackSource")),
  simVertexSrc_(config_.getParameter<edm::InputTag>("SimVertexSource")),
  otDigiToken_(consumes< edm::DetSetVector<Phase2TrackerDigi> >(otDigiSrc_)),
  otDigiSimLinkToken_(consumes< edm::DetSetVector<PixelDigiSimLink> >(otDigiSimLinkSrc_)),
  itPixelDigiToken_(consumes< edm::DetSetVector<PixelDigi> >(itPixelDigiSrc_)),
  itPixelDigiSimLinkToken_(consumes< edm::DetSetVector<PixelDigiSimLink> >(itPixelDigiSimLinkSrc_)),
  simTrackToken_(consumes< edm::SimTrackContainer >(simTrackSrc_)),
  simVertexToken_(consumes< edm::SimVertexContainer >(simVertexSrc_)),
  GeVperElectron(3.61E-09) // 1 electron(3.61eV, 1keV(277e, mod 9/06 d.k.
{
  for(auto itag : pSimHitSrc_) simHitTokens_.push_back(consumes< edm::PSimHitContainer >(itag));

  etaCut_  = config_.getParameter<double>("EtaCutOff");
  ptCut_  = config_.getParameter<double>("PtCutOff");
  edm::LogInfo("Phase2TrackerValidateDigi") << ">>> Construct Phase2TrackerValidateDigi ";
}

//
// destructor
//
Phase2TrackerValidateDigi::~Phase2TrackerValidateDigi() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  edm::LogInfo("Phase2TrackerValidateDigi")<< ">>> Destroy Phase2TrackerValidateDigi ";
}
//
// -- DQM Begin Run 
//
void Phase2TrackerValidateDigi::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
   edm::LogInfo("Phase2TrackerValidateDigi")<< "Initialize Phase2TrackerValidateDigi ";
}
//
// -- Analyze
//
void Phase2TrackerValidateDigi::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // Get digis
  iEvent.getByToken(itPixelDigiToken_ , itPixelDigiHandle_); 
  iEvent.getByToken(otDigiToken_ , otDigiHandle_); 

  // DigiSimLink
  iEvent.getByToken(itPixelDigiSimLinkToken_, itPixelSimLinkHandle_); 
  iEvent.getByToken(otDigiSimLinkToken_, otSimLinkHandle_); 

  // SimTrack
  iEvent.getByToken(simTrackToken_,simTracks);

  // SimVertex
  iEvent.getByToken(simVertexToken_,simVertices);

  // Tracker Topology 
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle_);

  std::vector<int> processTypes; 
  // Loop over Sim Tracks and Fill relevant histograms
  int nTracks = 0;
  int nTracksP = 0;
  int nTracksS = 0;
  for (edm::SimTrackContainer::const_iterator simTrkItr = simTracks->begin();
                                            simTrkItr != simTracks->end(); ++simTrkItr) {
    if (simTrkItr->charge() == 0) continue;
    int vtxIndex = simTrkItr->vertIndex();   
    int vtxParent = -1;
    if (vtxIndex > 0) {
      SimVertex vtx = (*simVertices)[vtxIndex];
      if (!vtx.noParent()) {
	int trkId = vtx.parentIndex();
        vtxParent = (*simTracks)[matchedSimTrack(simTracks, trkId)].vertIndex();
      } 
    } 
    int simTk_type = -1;
    //    if (vtxIndex == 0 || vtxParent == 0) type = isPrimary((*simTrkItr), simHits);
    if (vtxIndex == 0 || vtxParent == 0) simTk_type = 1;
    else simTk_type = 0;

    nTracks++; 
    if (simTk_type ==1) nTracksP++;
    else nTracksS++;

    float simTk_pt =  simTrkItr->momentum().pt();
    float simTk_eta = simTrkItr->momentum().eta();
    float simTk_phi = simTrkItr->momentum().phi();
    unsigned int simTk_id =  simTrkItr->trackId();

    
    if (fabs(simTk_eta) < etaCut_) fillHistogram(SimulatedTrackPt, SimulatedTrackPtP, SimulatedTrackPtS, simTk_pt, simTk_type);
    if (simTk_pt > ptCut_)  fillHistogram(SimulatedTrackEta, SimulatedTrackEtaP, SimulatedTrackEtaS, simTk_eta, simTk_type);
    if (fabs(simTk_eta) < etaCut_ && simTk_pt > ptCut_ ) fillHistogram(SimulatedTrackPhi, SimulatedTrackPhiP, SimulatedTrackPhiS, simTk_phi, simTk_type);
    
    edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
    if (theTkDigiGeomWatcher.check(iSetup)) {
      edm::ESHandle<TrackerGeometry> geomHandle;
      iSetup.get<TrackerDigiGeometryRecord>().get(geomType_, geomHandle);
      // initialize
      for (std::map<unsigned int, DigiMEs>::iterator it = layerMEs.begin(); it != layerMEs.end(); it++) {
	it->second.nDigis = 0;
	it->second.nHits  = 0;
      }
      fillSimHitInfo(iEvent, simTk_id, simTk_pt, simTk_eta, simTk_phi, simTk_type, geomHandle);
      int nHitCutoff = 2;
      if (pixelFlag_) nHitCutoff = 1;
      for (std::map<unsigned int, DigiMEs>::iterator it = layerMEs.begin(); it != layerMEs.end(); it++) {
	DigiMEs& local_mes = it->second;
	if (it->second.nHits < nHitCutoff) {
	  if (std::fabs(simTk_eta) < 1.0) local_mes.MissedHitTrackPt->Fill(simTk_pt);
	  if (simTk_pt > ptCut_ && std::fabs(simTk_eta) < 1.0 ) local_mes.MissedHitTrackEta->Fill(simTk_eta);
	  if (std::fabs(simTk_eta) < 1.0 && simTk_pt > ptCut_)  local_mes.MissedHitTrackPhi->Fill(simTk_phi);
	}
      }
    }
  }
  nSimulatedTracks->Fill(nTracks);
  nSimulatedTracksP->Fill(nTracksP);
  nSimulatedTracksS->Fill(nTracksS);
  if (pixelFlag_) fillITPixelBXInfo();
  else fillOTBXInfo();
}
  
int Phase2TrackerValidateDigi::fillSimHitInfo(const edm::Event& iEvent, unsigned int id, float pt, float eta, float phi, int type, const edm::ESHandle<TrackerGeometry> gHandle) {
  const TrackerTopology* tTopo = tTopoHandle_.product();
  const TrackerGeometry* tGeom = gHandle.product();  

  int totalHits = 0;
  for (auto& itoken : simHitTokens_) {
    edm::Handle<edm::PSimHitContainer> simHitHandle;
    iEvent.getByToken(itoken, simHitHandle);
    if (!simHitHandle.isValid()) continue;
    const edm::PSimHitContainer& simHits = (*simHitHandle.product());
    for(edm::PSimHitContainer::const_iterator isim = simHits.begin(); isim != simHits.end(); ++isim){
      if ((*isim).trackId() != id) continue; 
      unsigned int rawid = (*isim).detUnitId();
      int layer;
      if (pixelFlag_) layer = tTopo->getITPixelLayerNumber(rawid);
      else layer = tTopo->getOTLayerNumber(rawid);
      if (layer < 0) continue;

      std::map<unsigned int, DigiMEs>::iterator pos = layerMEs.find(layer);
      if (pos == layerMEs.end()) continue;
      DigiMEs& local_mes = pos->second;

      const DetId detId(rawid);
      float dZ = (*isim).entryPoint().z() - (*isim).exitPoint().z();  
      if (fabs(dZ) <= 0.01) continue;
      
      if (DetId(detId).det() != DetId::Detector::Tracker) continue;
        
      const GeomDet *geomDet = tGeom->idToDet(detId);
      if (!geomDet) continue;
      Global3DPoint pdPos = geomDet->surface().toGlobal(isim->localPosition());
      SimulatedXYPositionMap->Fill(pdPos.x()*10., pdPos.y()*10.);   
      SimulatedRZPositionMap->Fill(pdPos.z()*10., std::hypot(pdPos.x(),pdPos.y())*10.);   
      
      const TrackerGeomDet* geomDetUnit(tGeom->idToDetUnit(detId));
      const Phase2TrackerGeomDetUnit* tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(geomDetUnit);
      int nColumns  = tkDetUnit->specificTopology().ncolumns();
      
      totalHits++;
      pos->second.nHits++;
      
      
      local_mes.SimHitDx->Fill(std::fabs((*isim).entryPoint().x()-(*isim).exitPoint().x()));
      local_mes.SimHitDy->Fill(std::fabs((*isim).entryPoint().y()-(*isim).exitPoint().y()));
      local_mes.SimHitDz->Fill(std::fabs((*isim).entryPoint().z()-(*isim).exitPoint().z()));
      
      SimulatedTOFEtaMap->Fill(pdPos.eta(),(*isim).timeOfFlight());
      SimulatedTOFPhiMap->Fill(pdPos.phi(),(*isim).timeOfFlight());
      SimulatedTOFRMap->Fill(std::hypot(pdPos.x(),pdPos.y()),(*isim).timeOfFlight());
      SimulatedTOFZMap->Fill(pdPos.z(),(*isim).timeOfFlight());

      bool digiFlag;
      
      if (pixelFlag_) digiFlag = findITPixelDigi(rawid,id);
      else digiFlag = findOTDigi(rawid,id);

      if (fabs(eta) <  etaCut_) {
	local_mes.SimTrackPt->Fill(pt);
	if (digiFlag) local_mes.MatchedTrackPt->Fill(pt);
        else  local_mes.MissedDigiTrackPt->Fill(pt);
      }
      if (pt > ptCut_) {
	local_mes.SimTrackEta->Fill(eta);
        if (digiFlag) local_mes.MatchedTrackEta->Fill(eta);
        else local_mes.MissedDigiTrackEta->Fill(eta);
      }
      if ( fabs(eta) < etaCut_ && pt > ptCut_) {
	local_mes.SimTrackPhi->Fill(phi);
	if (digiFlag) local_mes.MatchedTrackPhi->Fill(phi);
        else local_mes.MissedDigiTrackPhi->Fill(phi);
      }
      if (digiFlag) {
        pos->second.nDigis++;
	MatchedRZPositionMap->Fill(pdPos.z()*10., std::hypot(pdPos.x(),pdPos.y())*10.);   
	MatchedXYPositionMap->Fill(pdPos.x()*10., pdPos.y()*10.);	
	if (nColumns <= 2) local_mes.MatchedSimHitElossS->Fill((*isim).energyLoss()/GeVperElectron);
	else local_mes.MatchedSimHitElossP->Fill((*isim).energyLoss()/GeVperElectron);
      } else {
	local_mes.MissedDigiLocalXposVsYPos->Fill((*isim).localPosition().x(), (*isim).localPosition().y());
        local_mes.MissedDigiTimeWindow->Fill(std::fabs((*isim).timeOfFlight() - pdPos.mag()/30.));
	if (nColumns <= 2) local_mes.MissedDigiSimHitElossS->Fill((*isim).energyLoss()/GeVperElectron);
	else local_mes.MissedDigiSimHitElossP->Fill((*isim).energyLoss()/GeVperElectron);

      }
    }
  }
  return totalHits;
}
bool Phase2TrackerValidateDigi::findOTDigi(unsigned int detid, unsigned int id) {
  bool matched = false;
  const edm::DetSetVector<Phase2TrackerDigi>* digis = otDigiHandle_.product();
  const edm::DetSetVector<PixelDigiSimLink>* links = otSimLinkHandle_.product();
  edm::DetSetVector<Phase2TrackerDigi>::const_iterator DSVIter = digis->find(detid);
  if (DSVIter != digis->end()) { 
    for (edm::DetSet<Phase2TrackerDigi>::const_iterator di = DSVIter->begin(); di != DSVIter->end(); di++) {   
      int col = di->column(); // column 
      int row = di->row();    // row
      unsigned int channel = Phase2TrackerDigi::pixelToChannel(row,col);      
      unsigned int simTkId = getSimTrackId(links, detid, channel);
      if (simTkId == id) {
	matched = true;
	break;
      } 
    }
  }
  return matched;
}
bool Phase2TrackerValidateDigi::findITPixelDigi(unsigned int detid, unsigned int id) {

  bool matched = false;
  const edm::DetSetVector<PixelDigi>* digis = itPixelDigiHandle_.product();
  const edm::DetSetVector<PixelDigiSimLink>* links = itPixelSimLinkHandle_.product();

  edm::DetSetVector<PixelDigi>::const_iterator DSVIter = digis->find(detid);
  if (DSVIter != digis->end()) { 
    for (edm::DetSet<PixelDigi>::const_iterator di = DSVIter->begin(); di != DSVIter->end(); di++) {   
      int col = di->column(); // column 
      int row = di->row();    // row
      unsigned int channel = PixelDigi::pixelToChannel(row,col);
      unsigned int simTkId = getSimTrackId(links, detid, channel);
      if (simTkId == id) {
	matched = true;
	break;
      } 
    }
  }
  return matched;
}
//
// -- Book Histograms
//
void Phase2TrackerValidateDigi::bookHistograms(DQMStore::IBooker & ibooker,
		 edm::Run const &  iRun ,
		 edm::EventSetup const &  iSetup ) {

  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  std::stringstream folder_name;

  ibooker.cd();
  folder_name << top_folder << "/" << "SimTrackInfo" ;
  ibooker.setCurrentFolder(folder_name.str());
 
  edm::LogInfo("Phase2TrackerValidateDigi")<< " Booking Histograms in : " << folder_name.str();
  std::stringstream HistoName;


  HistoName.str("");
  HistoName << "NumberOfSimulatedTracks";   
  nSimulatedTracks = ibooker.book1D(HistoName.str(),HistoName.str(), 501, -0.5, 500.5);

  HistoName.str("");
  HistoName << "NumberOfSimulatedTracksP";   
  nSimulatedTracksP = ibooker.book1D(HistoName.str(),HistoName.str(), 501, -0.5, 500.5);

  HistoName.str("");
  HistoName << "NumberOfSimulatedTracksS";   
  nSimulatedTracksS = ibooker.book1D(HistoName.str(),HistoName.str(), 501, -0.5, 500.5);

  edm::ParameterSet Parameters =  config_.getParameter<edm::ParameterSet>("TrackPtH");
  HistoName.str("");
  HistoName << "SimulatedTrackPt";   
  SimulatedTrackPt = ibooker.book1D(HistoName.str(),HistoName.str(),
				  Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
  
  HistoName.str("");
  HistoName << "SimulatedTrackPt";   
  SimulatedTrackPt = ibooker.book1D(HistoName.str(),HistoName.str(),
				  Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
  HistoName.str("");
  HistoName << "SimulatedTrackPtP";   
  SimulatedTrackPtP = ibooker.book1D(HistoName.str(),HistoName.str(),
				   Parameters.getParameter<int32_t>("Nbins"),
				   Parameters.getParameter<double>("xmin"),
				   Parameters.getParameter<double>("xmax"));
  HistoName.str("");
  HistoName << "SimulatedTrackPtS";   
  SimulatedTrackPtS = ibooker.book1D(HistoName.str(),HistoName.str(),
				   Parameters.getParameter<int32_t>("Nbins"),
				   Parameters.getParameter<double>("xmin"),
				   Parameters.getParameter<double>("xmax"));
  
  Parameters =  config_.getParameter<edm::ParameterSet>("TrackEtaH");

  HistoName.str("");
  HistoName << "SimulatedTrackEta"; 
  SimulatedTrackEta = ibooker.book1D(HistoName.str(),HistoName.str(),
				   Parameters.getParameter<int32_t>("Nbins"),
				   Parameters.getParameter<double>("xmin"),
				   Parameters.getParameter<double>("xmax"));
  HistoName.str("");
  HistoName << "SimulatedTrackEtaP";
  SimulatedTrackEtaP = ibooker.book1D(HistoName.str(),HistoName.str(),
				    Parameters.getParameter<int32_t>("Nbins"),
				    Parameters.getParameter<double>("xmin"),
				    Parameters.getParameter<double>("xmax"));
  
  HistoName.str("");
  HistoName << "SimulatedTrackEtaS";
  SimulatedTrackEtaS = ibooker.book1D(HistoName.str(),HistoName.str(),
				    Parameters.getParameter<int32_t>("Nbins"),
				    Parameters.getParameter<double>("xmin"),
				    Parameters.getParameter<double>("xmax"));
  
  Parameters =  config_.getParameter<edm::ParameterSet>("TrackPhiH");
  HistoName.str("");
  HistoName << "SimulatedTrackPhi";
  SimulatedTrackPhi = ibooker.book1D(HistoName.str(),HistoName.str(),
				   Parameters.getParameter<int32_t>("Nbins"),
				   Parameters.getParameter<double>("xmin"),
				   Parameters.getParameter<double>("xmax"));
  HistoName.str("");
  HistoName << "SimulatedTrackPhiP";
  SimulatedTrackPhiP = ibooker.book1D(HistoName.str(),HistoName.str(),
				    Parameters.getParameter<int32_t>("Nbins"),
				    Parameters.getParameter<double>("xmin"),
				    Parameters.getParameter<double>("xmax"));
  
  HistoName.str("");
  HistoName << "SimulatedTrackPhiS";
  SimulatedTrackPhiS = ibooker.book1D(HistoName.str(),HistoName.str(),
				    Parameters.getParameter<int32_t>("Nbins"),
				    Parameters.getParameter<double>("xmin"),
				    Parameters.getParameter<double>("xmax"));
  

  Parameters =  config_.getParameter<edm::ParameterSet>("XYPositionMapH");  
  HistoName.str("");
  HistoName << "SimulatedXPosVsYPos";
  SimulatedXYPositionMap = ibooker.book2D(HistoName.str(), HistoName.str(),
				 Parameters.getParameter<int32_t>("Nxbins"),
				 Parameters.getParameter<double>("xmin"),
				 Parameters.getParameter<double>("xmax"),
				 Parameters.getParameter<int32_t>("Nybins"),
				 Parameters.getParameter<double>("ymin"),
				 Parameters.getParameter<double>("ymax"));
  Parameters =  config_.getParameter<edm::ParameterSet>("RZPositionMapH");  
  HistoName.str("");
  HistoName << "SimulatedRPosVsZPos";
  SimulatedRZPositionMap = ibooker.book2D(HistoName.str(), HistoName.str(),
				 Parameters.getParameter<int32_t>("Nxbins"),
				 Parameters.getParameter<double>("xmin"),
				 Parameters.getParameter<double>("xmax"),
				 Parameters.getParameter<int32_t>("Nybins"),
				 Parameters.getParameter<double>("ymin"),
				 Parameters.getParameter<double>("ymax"));

  //add TOF maps
  Parameters =  config_.getParameter<edm::ParameterSet>("TOFEtaMapH");
  HistoName.str("");
  HistoName << "SimulatedTOFVsEta";
  SimulatedTOFEtaMap = ibooker.book2D(HistoName.str(), HistoName.str(),
					  Parameters.getParameter<int32_t>("Nxbins"),
					  Parameters.getParameter<double>("xmin"),
					  Parameters.getParameter<double>("xmax"),
					  Parameters.getParameter<int32_t>("Nybins"),
					  Parameters.getParameter<double>("ymin"),
					  Parameters.getParameter<double>("ymax"));

  Parameters =  config_.getParameter<edm::ParameterSet>("TOFPhiMapH");
  HistoName.str("");
  HistoName << "SimulatedTOFVsPhi";
  SimulatedTOFPhiMap = ibooker.book2D(HistoName.str(), HistoName.str(),
				      Parameters.getParameter<int32_t>("Nxbins"),
				      Parameters.getParameter<double>("xmin"),
				      Parameters.getParameter<double>("xmax"),
				      Parameters.getParameter<int32_t>("Nybins"),
				      Parameters.getParameter<double>("ymin"),
				      Parameters.getParameter<double>("ymax"));
  
  Parameters =  config_.getParameter<edm::ParameterSet>("TOFRMapH");
  HistoName.str("");
  HistoName << "SimulatedTOFVsR";
  SimulatedTOFRMap = ibooker.book2D(HistoName.str(), HistoName.str(),
                                      Parameters.getParameter<int32_t>("Nxbins"),
                                      Parameters.getParameter<double>("xmin"),
                                      Parameters.getParameter<double>("xmax"),
                                      Parameters.getParameter<int32_t>("Nybins"),
                                      Parameters.getParameter<double>("ymin"),
                                      Parameters.getParameter<double>("ymax"));

  Parameters =  config_.getParameter<edm::ParameterSet>("TOFZMapH");
  HistoName.str("");
  HistoName << "SimulatedTOFVsZ";
  SimulatedTOFZMap = ibooker.book2D(HistoName.str(), HistoName.str(),
                                      Parameters.getParameter<int32_t>("Nxbins"),
                                      Parameters.getParameter<double>("xmin"),
                                      Parameters.getParameter<double>("xmax"),
                                      Parameters.getParameter<int32_t>("Nybins"),
                                      Parameters.getParameter<double>("ymin"),
                                      Parameters.getParameter<double>("ymax"));

  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;

  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle_);
  const TrackerTopology* const tTopo = tTopoHandle_.product();

  if (theTkDigiGeomWatcher.check(iSetup)) {
    edm::ESHandle<TrackerGeometry> geom_handle;
    iSetup.get<TrackerDigiGeometryRecord>().get(geomType_, geom_handle);
    const TrackerGeometry* tGeom = geom_handle.product();  
    
    for (auto const & det_u : tGeom->detUnits()) {
      unsigned int detId_raw = det_u->geographicalId().rawId();
      bookLayerHistos(ibooker,detId_raw, tTopo,pixelFlag_); 
    }
  }
  ibooker.cd();
  folder_name.str("");
  folder_name << top_folder << "/" << "DigiMonitor";
  ibooker.setCurrentFolder(folder_name.str());

  Parameters =  config_.getParameter<edm::ParameterSet>("XYPositionMapH");  
  HistoName.str("");
  HistoName << "MatchedSimXPosVsYPos";
  MatchedXYPositionMap = ibooker.book2D(HistoName.str(), HistoName.str(),
				 Parameters.getParameter<int32_t>("Nxbins"),
				 Parameters.getParameter<double>("xmin"),
				 Parameters.getParameter<double>("xmax"),
				 Parameters.getParameter<int32_t>("Nybins"),
				 Parameters.getParameter<double>("ymin"),
				 Parameters.getParameter<double>("ymax"));
  Parameters =  config_.getParameter<edm::ParameterSet>("RZPositionMapH");  
  HistoName.str("");
  HistoName << "MatchedSimRPosVsZPos";
  MatchedRZPositionMap = ibooker.book2D(HistoName.str(), HistoName.str(),
				 Parameters.getParameter<int32_t>("Nxbins"),
				 Parameters.getParameter<double>("xmin"),
				 Parameters.getParameter<double>("xmax"),
				 Parameters.getParameter<int32_t>("Nybins"),
				 Parameters.getParameter<double>("ymin"),
				 Parameters.getParameter<double>("ymax"));

}
//
// -- Book Layer Histograms
//
void Phase2TrackerValidateDigi::bookLayerHistos(DQMStore::IBooker & ibooker, unsigned int det_id, const TrackerTopology* tTopo, bool flag){ 

  int layer;
  if (flag) layer = tTopo->getITPixelLayerNumber(det_id);
  else layer = tTopo->getOTLayerNumber(det_id);

  if (layer < 0) return;
  std::map<uint32_t, DigiMEs >::iterator pos = layerMEs.find(layer);
  if (pos == layerMEs.end()) {

    std::string top_folder = config_.getParameter<std::string>("TopFolderName");
    std::stringstream folder_name;

    std::ostringstream fname1, fname2, tag;
    if (layer < 100) { 
      fname1 << "Barrel";
      fname2 << "Layer_" << layer;    
    } else {
      int side = layer/100;
      int idisc = layer - side*100; 
      fname1 << "EndCap_Side_" << side; 
      fname2 << "Disc_" << idisc;       
    }
   
    ibooker.cd();
    folder_name << top_folder << "/" << "DigiMonitor" << "/"<< fname1.str() << "/" << fname2.str() ;
    edm::LogInfo("Phase2TrackerValidateDigi")<< " Booking Histograms in : " << folder_name.str();

    ibooker.setCurrentFolder(folder_name.str());

    std::ostringstream HistoName;    


    DigiMEs local_mes;

    edm::ParameterSet Parameters =  config_.getParameter<edm::ParameterSet>("TrackPtH");
    HistoName.str("");
    HistoName << "SimTrackPt_" << fname2.str();   
    local_mes.SimTrackPt = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackPt_" << fname2.str();   
    local_mes.MatchedTrackPt = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MissedHitTrackPt_" << fname2.str();   
    local_mes.MissedHitTrackPt = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MissedDigiTrackPt_" << fname2.str();   
    local_mes.MissedDigiTrackPt = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));

    Parameters =  config_.getParameter<edm::ParameterSet>("TrackEtaH");

    HistoName.str("");
    HistoName << "SimTrackEta_" << fname2.str();   
    local_mes.SimTrackEta = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackEta_" << fname2.str();   
    local_mes.MatchedTrackEta = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));

    HistoName.str("");
    HistoName << "MissedHitTrackEta_" << fname2.str();   
    local_mes.MissedHitTrackEta = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));

    HistoName.str("");
    HistoName << "MissedDigiTrackEta_" << fname2.str();   
    local_mes.MissedDigiTrackEta = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));

    Parameters =  config_.getParameter<edm::ParameterSet>("TrackPhiH");

    HistoName.str("");
    HistoName << "SimTrackPhi_" << fname2.str();   
    local_mes.SimTrackPhi = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackPhi_" << fname2.str();   
    local_mes.MatchedTrackPhi = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MissedHitTrackPhi_" << fname2.str();   
    local_mes.MissedHitTrackPhi = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MissedDigiTrackPhi_" << fname2.str();   
    local_mes.MissedDigiTrackPhi = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));

    Parameters =  config_.getParameter<edm::ParameterSet>("SimHitElossH");
    HistoName.str("");
    HistoName << "MatchedSimHitElossS_" << fname2.str();   
    local_mes.MatchedSimHitElossS = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));

    HistoName.str("");
    HistoName << "MatchedSimHitElossP_" << fname2.str();   
    local_mes.MatchedSimHitElossP = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));

    HistoName.str("");
    HistoName << "MissedDigiSimHitElossS_" << fname2.str();   
    local_mes.MissedDigiSimHitElossS = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MissedDigiSimHitElossP_" << fname2.str();   
    local_mes.MissedDigiSimHitElossP = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));

    Parameters =  config_.getParameter<edm::ParameterSet>("SimHitDxH");
    HistoName.str("");
    HistoName << "SimHitDx_" << fname2.str();
    local_mes.SimHitDx = ibooker.book1D(HistoName.str(),HistoName.str(),
					Parameters.getParameter<int32_t>("Nbins"),
					Parameters.getParameter<double>("xmin"),
					Parameters.getParameter<double>("xmax"));
    
    Parameters =  config_.getParameter<edm::ParameterSet>("SimHitDyH");
    HistoName.str("");
    HistoName << "SimHitDy_" << fname2.str();
    local_mes.SimHitDy = ibooker.book1D(HistoName.str(),HistoName.str(),
					 Parameters.getParameter<int32_t>("Nbins"),
					 Parameters.getParameter<double>("xmin"),
					 Parameters.getParameter<double>("xmax"));

    Parameters =  config_.getParameter<edm::ParameterSet>("SimHitDzH");
    HistoName.str("");
    HistoName << "SimHitDz_" << fname2.str();
    local_mes.SimHitDz = ibooker.book1D(HistoName.str(),HistoName.str(),
					 Parameters.getParameter<int32_t>("Nbins"),
					 Parameters.getParameter<double>("xmin"),
					 Parameters.getParameter<double>("xmax"));
    
    HistoName.str("");
    HistoName << "BunchXingWindow_" << fname2.str();
    local_mes.BunchXTimeBin = ibooker.book1D(HistoName.str(),HistoName.str(), 8, -5.5, 2.5);
    
    HistoName.str("");
    HistoName << "FractionOfOOTPUDigi_" << fname2.str();
    local_mes.FractionOfOOTDigis = ibooker.bookProfile(HistoName.str(),HistoName.str(), 8, -5.5, 2.5, 0., 1.0,"s");

    HistoName.str("");
    HistoName << "MissedDigiLocalXPosvsYPos_" << fname2.str();
    local_mes.MissedDigiLocalXposVsYPos = ibooker.book2D(HistoName.str(),HistoName.str(), 130, -6.5, 6.5, 130, -6.5, 6.5);

    Parameters =  config_.getParameter<edm::ParameterSet>("TOFEtaMapH");
    HistoName.str("");
    HistoName << "MissedDigiTimeWindow_"<< fname2.str();
    local_mes.MissedDigiTimeWindow = ibooker.book1D(HistoName.str(), HistoName.str(),100, -0.5, 49.5);

    local_mes.nDigis  = 0;
    layerMEs.insert(std::make_pair(layer, local_mes)); 
  }  
}
//
// -- Get SimTrack Id
//
unsigned int Phase2TrackerValidateDigi::getSimTrackId(const edm::DetSetVector<PixelDigiSimLink> * simLinks, const DetId& detId, unsigned int& channel) {

  edm::DetSetVector<PixelDigiSimLink>::const_iterator 
    isearch = simLinks->find(detId);

  unsigned int simTrkId(0);
  if (isearch == simLinks->end()) return simTrkId;

  edm::DetSet<PixelDigiSimLink> link_detset = (*simLinks)[detId];
  // Loop over DigiSimLink in this det unit
  int iSimLink = 0;
  for (edm::DetSet<PixelDigiSimLink>::const_iterator it = link_detset.data.begin(); it != link_detset.data.end(); it++,iSimLink++) {
    if (channel == it->channel()) {
      simTrkId = it->SimTrackId();
      break;        
    } 
  }
  return simTrkId;
}
void Phase2TrackerValidateDigi::fillOTBXInfo() {

  const edm::DetSetVector<PixelDigiSimLink>* links = otSimLinkHandle_.product();

  for (typename edm::DetSetVector<PixelDigiSimLink>::const_iterator DSViter = links->begin(); DSViter != links->end(); DSViter++) {
    unsigned int rawid = DSViter->id; 
    DetId detId(rawid);
    if (DetId(detId).det() != DetId::Detector::Tracker) continue;
    int layer = tTopoHandle_->getOTLayerNumber(rawid);
    if (layer < 0) continue;
    std::map<uint32_t, DigiMEs >::iterator pos = layerMEs.find(layer);
    if (pos == layerMEs.end()) continue;
    DigiMEs& local_mes = pos->second;
    int tot_digi = 0;
    std::map<int, float> bxMap;
    for (typename edm::DetSet< PixelDigiSimLink >::const_iterator di = DSViter->begin(); di != DSViter->end(); di++) {
      tot_digi++;
      int bx = di->eventId().bunchCrossing();
      std::map<int, float>::iterator ic = bxMap.find(bx);
      if (ic == bxMap.end()) bxMap[bx] = 1.0;
      else bxMap[bx] += 1.0;
    }
    for (auto & v : bxMap) {
      if (tot_digi) {
	local_mes.BunchXTimeBin->Fill(v.first, v.second);
	local_mes.FractionOfOOTDigis->Fill(v.first,v.second/tot_digi);
      }
    }
  }
}
void Phase2TrackerValidateDigi::fillITPixelBXInfo() {

  const edm::DetSetVector<PixelDigiSimLink>* links = itPixelSimLinkHandle_.product();

  for (typename edm::DetSetVector<PixelDigiSimLink>::const_iterator DSViter = links->begin(); DSViter != links->end(); DSViter++) {
    unsigned int rawid = DSViter->id; 
    DetId detId(rawid);
    if (DetId(detId).det() != DetId::Detector::Tracker) continue;
    int layer = tTopoHandle_->getITPixelLayerNumber(rawid);
    if (layer < 0) continue;
    std::map<uint32_t, DigiMEs >::iterator pos = layerMEs.find(layer);
    if (pos == layerMEs.end()) continue;
    DigiMEs& local_mes = pos->second;
    int tot_digi = 0;
    std::map<int, float> bxMap;
    for (typename edm::DetSet< PixelDigiSimLink >::const_iterator di = DSViter->begin(); di != DSViter->end(); di++) {
      tot_digi++;
      int bx = di->eventId().bunchCrossing();
      std::map<int, float>::iterator ic = bxMap.find(bx);
      if (ic == bxMap.end()) bxMap[bx] = 1.0;
      else bxMap[bx] += 1.0;
    }
    for (auto & v : bxMap) {
      if (tot_digi) {
	local_mes.BunchXTimeBin->Fill(v.first, v.second);
	local_mes.FractionOfOOTDigis->Fill(v.first,v.second/tot_digi);
      }
    }
  }
}
//
// -- Get Matched SimTrack  
//
int Phase2TrackerValidateDigi::matchedSimTrack(edm::Handle<edm::SimTrackContainer>& SimTk, unsigned int simTrkId) {

  edm::SimTrackContainer sim_tracks = (*SimTk.product());
  for(unsigned int it = 0; it < sim_tracks.size(); it++) {
    if (sim_tracks[it].trackId() == simTrkId) {
      return it;
    }
  }
  return -1;
}
//
//  -- Check if the SimTrack is _Primary or not 
//
int Phase2TrackerValidateDigi::isPrimary(const SimTrack& simTrk, edm::Handle<edm::PSimHitContainer>& simHitHandle) {
  int result = -1;
  unsigned int trkId = simTrk.trackId();
  int vtxIndx = simTrk.vertIndex();
  if (trkId > 0) {
    //    int vtxIndx = simTrk.vertIndex();
    const edm::PSimHitContainer& sim_hits = (*simHitHandle.product());
    for (edm::PSimHitContainer::const_iterator iHit = sim_hits.begin(); iHit != sim_hits.end(); ++iHit) {
      if (trkId == iHit->trackId()) {
	int ptype = iHit->processType();
	if (  (vtxIndx == 0 ) && (ptype == 2 || ptype == 7 || ptype == 9 || ptype == 11 || ptype == 13 ||ptype == 15) ) result = 1;
        else result = 0; 
	break;
      }
    }
  }
  return result;
}
//
// -- Fill HistogramSiStripTemplateDepFakeESSource
//
void Phase2TrackerValidateDigi::fillHistogram(MonitorElement* th1, MonitorElement* th2, MonitorElement* th3, float val, int primary){
  if (th1) th1->Fill(val);
  if (th2 && primary == 1) th2->Fill(val);
  if (th3 && primary != 1) th3->Fill(val);
}
//define this as a plug-in
DEFINE_FWK_MODULE(Phase2TrackerValidateDigi);
