// -*- C++ -*-
//
// Package:    Phase2TrackerValidateDigi
// Class:      Phase2TrackerValidateDigi
// 
/**\class Phase2TrackerValidateDigi Phase2TrackerValidateDigi.cc 

 Description: Test pixel digis. 

*/
//
// Author:  Suchandra Dutta
// Created:  November 2013
//
//
// system include files
#include <memory>
#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerValidateDigi.h"
#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerDigiCommon.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"


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
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

// // constructors 
//
Phase2TrackerValidateDigi::Phase2TrackerValidateDigi(const edm::ParameterSet& iConfig) :
  dqmStore_(edm::Service<DQMStore>().operator->()),
  config_(iConfig)
{
  digiSrc_ = config_.getParameter<edm::InputTag>("DigiSource");
  digiSimLinkSrc_ = config_.getParameter<edm::InputTag>("DigiSimLinkSource");
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
// -- Begin Job
//
void Phase2TrackerValidateDigi::beginJob() {
   edm::LogInfo("Phase2TrackerValidateDigi")<< "Initialize Phase2TrackerValidateDigi ";
}
//
// -- Begin Run
//
void Phase2TrackerValidateDigi::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup){
  bookHistos();
}
//
// -- Analyze
//
void Phase2TrackerValidateDigi::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // Get digis
  edm::Handle< edm::DetSetVector<Phase2TrackerDigi> > digiHandle;
  iEvent.getByLabel(digiSrc_, digiHandle);
  const DetSetVector<Phase2TrackerDigi>* digis = digiHandle.product();


  // DigiSimLink
  edm::Handle< edm::DetSetVector<PixelDigiSimLink> > simLinks;
  iEvent.getByLabel(digiSrc_,   simLinks);
  

  // PSimHits
  edm::Handle<edm::PSimHitContainer> simHits;
  iEvent.getByLabel("g4SimHits","TrackerHitsPixelBarrelLowTof" ,simHits);

  // SimTrack
  edm::Handle<edm::SimTrackContainer> simTracks;
  iEvent.getByLabel("g4SimHits",simTracks);

  // SimVertex
  edm::Handle<edm::SimVertexContainer> simVertices;
  iEvent.getByLabel("g4SimHits", simVertices);

  // Tracker Topology 
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
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
    unsigned int layer = phase2trackerdigi::getLayerNumber(rawid, tTopo);
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
    if (pos == layerMEs.end()) {
      bookLayerHistos(layer);
      pos = layerMEs.find(layer);
    }
    bool matched = false;
    edm::DetSetVector<Phase2TrackerDigi>::const_iterator DSVIter = digis->find(rawid);
    if (DSVIter != digis->end()) { 
      for (DetSet<Phase2TrackerDigi>::const_iterator di = DSVIter->begin(); di != DSVIter->end(); di++) {   
	int col = di->column(); // column 
	int row = di->row();    // row
	
	unsigned int channel = PixelChannelIdentifier::pixelToChannel(row,col);
	unsigned int simTkId = getSimTrackId(simLinks, detId, channel);
	if (simTkId == trackid) {
	  matched = true;
          break;
	} 
      }
    }
    DigiMEs local_mes = pos->second;
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
  void Phase2TrackerValidateDigi::bookHistos() {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  std::stringstream folder_name;

  dqmStore_->cd();
  folder_name << top_folder << "/" << "SimTrackInfo" ;
  dqmStore_->setCurrentFolder(folder_name.str());

  edm::LogInfo("Phase2TrackerValidateDigi")<< " Booking Histograms in : " << folder_name.str();
  std::stringstream HistoName;

  edm::ParameterSet Parameters =  config_.getParameter<edm::ParameterSet>("TrackPtH");
  HistoName << "SimulatedTrackPt";   
  SimulatedTrackPt = dqmStore_->book1D(HistoName.str(),HistoName.str(),
				  Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
  HistoName.str("");
  HistoName << "SimulatedTrackPtP";   
  SimulatedTrackPtP = dqmStore_->book1D(HistoName.str(),HistoName.str(),
				   Parameters.getParameter<int32_t>("Nbins"),
				   Parameters.getParameter<double>("xmin"),
				   Parameters.getParameter<double>("xmax"));
  HistoName.str("");
  HistoName << "SimulatedTrackPtS";   
  SimulatedTrackPtS = dqmStore_->book1D(HistoName.str(),HistoName.str(),
				   Parameters.getParameter<int32_t>("Nbins"),
				   Parameters.getParameter<double>("xmin"),
				   Parameters.getParameter<double>("xmax"));
  
  Parameters =  config_.getParameter<edm::ParameterSet>("TrackEtaH");

  HistoName.str("");
  HistoName << "SimulatedTrackEta"; 
  SimulatedTrackEta = dqmStore_->book1D(HistoName.str(),HistoName.str(),
				   Parameters.getParameter<int32_t>("Nbins"),
				   Parameters.getParameter<double>("xmin"),
				   Parameters.getParameter<double>("xmax"));
  HistoName.str("");
  HistoName << "SimulatedTrackEtaP";
  SimulatedTrackEtaP = dqmStore_->book1D(HistoName.str(),HistoName.str(),
				    Parameters.getParameter<int32_t>("Nbins"),
				    Parameters.getParameter<double>("xmin"),
				    Parameters.getParameter<double>("xmax"));
  
  HistoName.str("");
  HistoName << "SimulatedTrackEtaS";
  SimulatedTrackEtaS = dqmStore_->book1D(HistoName.str(),HistoName.str(),
				    Parameters.getParameter<int32_t>("Nbins"),
				    Parameters.getParameter<double>("xmin"),
				    Parameters.getParameter<double>("xmax"));
  
  Parameters =  config_.getParameter<edm::ParameterSet>("TrackPhiH");
  HistoName.str("");
  HistoName << "SimulatedTrackPhi";
  SimulatedTrackPhi = dqmStore_->book1D(HistoName.str(),HistoName.str(),
				   Parameters.getParameter<int32_t>("Nbins"),
				   Parameters.getParameter<double>("xmin"),
				   Parameters.getParameter<double>("xmax"));
  HistoName.str("");
  HistoName << "SiulatedmTrackPhiP";
  SimulatedTrackPhiP = dqmStore_->book1D(HistoName.str(),HistoName.str(),
				    Parameters.getParameter<int32_t>("Nbins"),
				    Parameters.getParameter<double>("xmin"),
				    Parameters.getParameter<double>("xmax"));
  
  HistoName.str("");
  HistoName << "SimulatedTrackPhiS";
  SimulatedTrackPhiS = dqmStore_->book1D(HistoName.str(),HistoName.str(),
				    Parameters.getParameter<int32_t>("Nbins"),
				    Parameters.getParameter<double>("xmin"),
				    Parameters.getParameter<double>("xmax"));
}
//
// -- Book Histograms
//
void Phase2TrackerValidateDigi::bookLayerHistos(unsigned int ilayer){ 
  std::map<uint32_t, DigiMEs >::iterator pos = layerMEs.find(ilayer);
  if (pos == layerMEs.end()) {

    std::string top_folder = config_.getParameter<std::string>("TopFolderName");
    std::stringstream folder_name;

    std::ostringstream fname1, fname2, tag;
    if (ilayer < 100) { 
      fname1 << "Barrel";
      fname2 << "Layer_" << ilayer;    
    } else {
      int side = ilayer/100;
      int idisc = ilayer - side*100; 
      fname1 << "EndCap_Side_" << side; 
      fname2 << "Disc_" << idisc;       
      std::cout << " Creating histograms for Disc " << idisc << " with " << fname2.str() << std::endl; 
    }
   
    dqmStore_->cd();
    folder_name << top_folder << "/" << "DigiMonitor" << "/"<< fname1.str() << "/" << fname2.str() ;
    edm::LogInfo("Phase2TrackerValidateDigi")<< " Booking Histograms in : " << folder_name.str();

    dqmStore_->setCurrentFolder(folder_name.str());

    std::ostringstream HistoName;    


    DigiMEs local_mes;

    edm::ParameterSet Parameters =  config_.getParameter<edm::ParameterSet>("TrackPtH");
    HistoName.str("");
    HistoName << "SimTrackPt_" << fname2.str();   
    local_mes.SimTrackPt = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "SimTrackPtP_" << fname2.str();   
    local_mes.SimTrackPtP = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "SimTrackPtS_" << fname2.str();   
    local_mes.SimTrackPtS = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackPt_" << fname2.str();   
    local_mes.MatchedTrackPt = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackPtP_" << fname2.str();   
    local_mes.MatchedTrackPtP = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackPtS_" << fname2.str();   
    local_mes.MatchedTrackPtS = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));

    Parameters =  config_.getParameter<edm::ParameterSet>("TrackEtaH");

    HistoName.str("");
    HistoName << "SimTrackEta_" << fname2.str();   
    local_mes.SimTrackEta = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "SimTrackEtaP_" << fname2.str();   
    local_mes.SimTrackEtaP = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "SimTrackEtaS_" << fname2.str();   
    local_mes.SimTrackEtaS = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackEta_" << fname2.str();   
    local_mes.MatchedTrackEta = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackEtaP_" << fname2.str();   
    local_mes.MatchedTrackEtaP = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackEtaS_" << fname2.str();   
    local_mes.MatchedTrackEtaS = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));

    Parameters =  config_.getParameter<edm::ParameterSet>("TrackPhiH");

    HistoName.str("");
    HistoName << "SimTrackPhi_" << fname2.str();   
    local_mes.SimTrackPhi = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "SimTrackPhiP_" << fname2.str();   
    local_mes.SimTrackPhiP = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "SimTrackPhiS_" << fname2.str();   
    local_mes.SimTrackPhiS = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackPhi_" << fname2.str();   
    local_mes.MatchedTrackPhi = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackPhiP_" << fname2.str();   
    local_mes.MatchedTrackPhiP = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    HistoName.str("");
    HistoName << "MatchedTrackPhiS_" << fname2.str();   
    local_mes.MatchedTrackPhiS = dqmStore_->book1D(HistoName.str(),HistoName.str(),
						Parameters.getParameter<int32_t>("Nbins"),
						Parameters.getParameter<double>("xmin"),
						Parameters.getParameter<double>("xmax"));
    layerMEs.insert(std::make_pair(ilayer, local_mes)); 
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
// -- Fill Histogram
//
void Phase2TrackerValidateDigi::fillHistogram(MonitorElement* th1, MonitorElement* th2, MonitorElement* th3, float val, int primary){
  if (th1 && th2 && th3) {
    th1->Fill(val);
    if (primary == 1) th2->Fill(val);
    else th3->Fill(val);
  }
}
//
// -- End Job
//
void Phase2TrackerValidateDigi::endJob(){
  dqmStore_->cd();
  dqmStore_->showDirStructure();  
}
//define this as a plug-in
DEFINE_FWK_MODULE(Phase2TrackerValidateDigi);
