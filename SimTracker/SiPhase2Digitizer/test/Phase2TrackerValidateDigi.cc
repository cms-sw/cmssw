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
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESWatcher.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
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

// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"

// // constructors 
//
Phase2TrackerValidateDigi::Phase2TrackerValidateDigi(const edm::ParameterSet& iConfig) :
  config_(iConfig),
  pixDigiSrc_(config_.getParameter<edm::InputTag>("PixelDigiSource")),
  otDigiSrc_(config_.getParameter<edm::InputTag>("OuterTrackerDigiSource")),
  digiSimLinkSrc_(config_.getParameter<edm::InputTag>("DigiSimLinkSource")),
  pSimHitSrc_(config_.getParameter<edm::InputTag>("PSimHitSource")),
  simTrackSrc_(config_.getParameter<edm::InputTag>("SimTrackSource")),
  simVertexSrc_(config_.getParameter<edm::InputTag>("SimVertexSource")),
  pixDigiToken_(consumes< edm::DetSetVector<PixelDigi> >(pixDigiSrc_)),
  otDigiToken_(consumes< edm::DetSetVector<Phase2TrackerDigi> >(otDigiSrc_)),
  digiSimLinkToken_(consumes< edm::DetSetVector<PixelDigiSimLink> >(digiSimLinkSrc_)),
  psimHitToken_(consumes< edm::PSimHitContainer >(pSimHitSrc_)),
  simTrackToken_(consumes< edm::SimTrackContainer >(simTrackSrc_)),
  simVertexToken_(consumes< edm::SimVertexContainer >(simVertexSrc_))
{
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
  edm::Handle< edm::DetSetVector<PixelDigi> > pixDigiHandle;
  iEvent.getByToken(pixDigiToken_ , pixDigiHandle); 

  edm::Handle< edm::DetSetVector<Phase2TrackerDigi> > otDigiHandle;
  iEvent.getByToken(otDigiToken_ , otDigiHandle); 
  const DetSetVector<Phase2TrackerDigi>* digis = otDigiHandle.product();


  // DigiSimLink
  edm::Handle< edm::DetSetVector<PixelDigiSimLink> > simLinks;
  iEvent.getByToken(digiSimLinkToken_ , simLinks); 

  // PSimHits
  edm::Handle<edm::PSimHitContainer> simHits;
  iEvent.getByToken(psimHitToken_,simHits);

  // SimTrack
  edm::Handle<edm::SimTrackContainer> simTracks;
  iEvent.getByToken(simTrackToken_,simTracks);

  // SimVertex
  edm::Handle<edm::SimVertexContainer> simVertices;
  iEvent.getByToken(simVertexToken_,simVertices);

  // Tracker Topology 
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* tTopo = tTopoHandle.product();

  std::vector<int> processTypes; 
  // Loop over Sim Tracks and Fill relevant histograms
  int nTracks = 0;
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
    int type = -1;
    if (vtxIndex == 0 || vtxParent == 0) type = isPrimary((*simTrkItr), simHits);

    processTypes.push_back(type);    
    nTracks++; 
    float simTk_pt =  simTrkItr->momentum().pt();
    float simTk_eta = simTrkItr->momentum().eta();
    float simTk_phi = simTrkItr->momentum().phi();

    fillHistogram(SimulatedTrackPt, SimulatedTrackPtP, SimulatedTrackPtS, simTk_pt, type);
    fillHistogram(SimulatedTrackEta, SimulatedTrackEtaP, SimulatedTrackEtaS, simTk_eta, type);
    fillHistogram(SimulatedTrackPhi, SimulatedTrackPhiP, SimulatedTrackPhiS, simTk_phi, type);
  }

  for(std::vector<PSimHit>::const_iterator isim = simHits->begin(); isim != simHits->end(); ++isim){
    unsigned int rawid = (*isim).detUnitId();
    int layer = tTopo->getOTLayerNumber(rawid);
    if (layer < 0) continue;
    const DetId detId(rawid);
    unsigned int trackid = (*isim).trackId();
    int iSimTrk = matchedSimTrack(simTracks, trackid);
    // remove hits from neutrals
    if (iSimTrk < 0 || (*simTracks)[iSimTrk].charge() == 0) continue;
    float dZ = (*isim).entryPoint().z() - (*isim).exitPoint().z();  
    if (fabs(dZ) <= 0.01) continue;

    float simTk_pt  = (*simTracks)[iSimTrk].momentum().pt();
    float simTk_eta = (*simTracks)[iSimTrk].momentum().eta();
    float simTk_phi = (*simTracks)[iSimTrk].momentum().phi();
    int simTk_type = processTypes[iSimTrk];

    std::map<unsigned int, DigiMEs>::iterator pos = layerMEs.find(layer);
    if (pos == layerMEs.end()) continue;
    DigiMEs local_mes = pos->second;

    bool matched = false;
    edm::DetSetVector<Phase2TrackerDigi>::const_iterator DSVIter = digis->find(rawid);
    if (DSVIter != digis->end()) { 
      for (DetSet<Phase2TrackerDigi>::const_iterator di = DSVIter->begin(); di != DSVIter->end(); di++) {   
	int col = di->column(); // column 
	int row = di->row();    // row
	
	unsigned int channel = Phase2TrackerDigi::pixelToChannel(row,col);
	unsigned int simTkId = getSimTrackId(simLinks, detId, channel);
	if (simTkId == trackid) {
	  matched = true;
          break;
	} 
      }
    }
    if (fabs(simTk_eta) <= etaCut_) {
      fillHistogram(local_mes.SimTrackPt, local_mes.SimTrackPtP, local_mes.SimTrackPtS, simTk_pt, simTk_type);
      if (matched) fillHistogram(local_mes.MatchedTrackPt, local_mes.MatchedTrackPtP, local_mes.MatchedTrackPtS, simTk_pt, simTk_type);
    }
    if (simTk_pt > ptCut_) {
      fillHistogram(local_mes.SimTrackEta, local_mes.SimTrackEtaP, local_mes.SimTrackEtaS, simTk_eta, simTk_type);
      if (matched) fillHistogram(local_mes.MatchedTrackEta, local_mes.MatchedTrackEtaP, local_mes.MatchedTrackEtaS, simTk_eta, simTk_type);
    }
    if (simTk_pt > ptCut_ && fabs(simTk_eta) <= etaCut_) {
      fillHistogram(local_mes.SimTrackPhi, local_mes.SimTrackPhiP, local_mes.SimTrackPhiS, simTk_phi, simTk_type);
      if (matched) fillHistogram(local_mes.MatchedTrackPhi, local_mes.MatchedTrackPhiP, local_mes.MatchedTrackPhiS, simTk_phi, simTk_type);
    }  
  }
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

  edm::ParameterSet Parameters =  config_.getParameter<edm::ParameterSet>("TrackPtH");
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
  HistoName << "SiulatedmTrackPhiP";
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

  std::string geometry_type = config_.getParameter<std::string>("GeometryType");
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;

  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  if (theTkDigiGeomWatcher.check(iSetup)) {
    edm::ESHandle<TrackerGeometry> geom_handle;
    iSetup.get<TrackerDigiGeometryRecord>().get(geometry_type, geom_handle);
    for (auto const & det_u : geom_handle->detUnits()) {
      unsigned int detId_raw = det_u->geographicalId().rawId();
      bookLayerHistos(ibooker,detId_raw, tTopo); 
    }
  }
  ibooker.cd();
  ibooker.setCurrentFolder(folder_name.str());
}
//
// -- Book Layer Histograms
//
void Phase2TrackerValidateDigi::bookLayerHistos(DQMStore::IBooker & ibooker, unsigned int det_id, const TrackerTopology* tTopo){ 

  int layer = tTopo->getOTLayerNumber(det_id);
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
    HistoName << "SimTrackPtP_" << fname2.str();   
    local_mes.SimTrackPtP = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "SimTrackPtS_" << fname2.str();   
    local_mes.SimTrackPtS = ibooker.book1D(HistoName.str(),HistoName.str(),
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
    HistoName << "MatchedTrackPtP_" << fname2.str();   
    local_mes.MatchedTrackPtP = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackPtS_" << fname2.str();   
    local_mes.MatchedTrackPtS = ibooker.book1D(HistoName.str(),HistoName.str(),
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
    HistoName << "SimTrackEtaP_" << fname2.str();   
    local_mes.SimTrackEtaP = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "SimTrackEtaS_" << fname2.str();   
    local_mes.SimTrackEtaS = ibooker.book1D(HistoName.str(),HistoName.str(),
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
    HistoName << "MatchedTrackEtaP_" << fname2.str();   
    local_mes.MatchedTrackEtaP = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackEtaS_" << fname2.str();   
    local_mes.MatchedTrackEtaS = ibooker.book1D(HistoName.str(),HistoName.str(),
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
    HistoName << "SimTrackPhiP_" << fname2.str();   
    local_mes.SimTrackPhiP = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "SimTrackPhiS_" << fname2.str();   
    local_mes.SimTrackPhiS = ibooker.book1D(HistoName.str(),HistoName.str(),
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
    HistoName << "MatchedTrackPhiP_" << fname2.str();   
    local_mes.MatchedTrackPhiP = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackPhiS_" << fname2.str();   
    local_mes.MatchedTrackPhiS = ibooker.book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    layerMEs.insert(std::make_pair(layer, local_mes)); 
  }  
}
//
// -- Get SimTrack Id
//
unsigned int Phase2TrackerValidateDigi::getSimTrackId(edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& simLinks, const DetId& detId, unsigned int& channel) {

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
int Phase2TrackerValidateDigi::isPrimary(const SimTrack& simTrk, edm::Handle<edm::PSimHitContainer>& simHits) {
  int result = -1;
  unsigned int trkId = simTrk.trackId();
  int vtxIndx = simTrk.vertIndex();
  if (trkId > 0) {
    //    int vtxIndx = simTrk.vertIndex();
    for (edm::PSimHitContainer::const_iterator iHit = simHits->begin(); iHit != simHits->end(); ++iHit) {
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
  if (th1 && th2 && th3) {
    th1->Fill(val);
    if (primary == 1) th2->Fill(val);
    else th3->Fill(val);
  }
}
//define this as a plug-in
DEFINE_FWK_MODULE(Phase2TrackerValidateDigi);
