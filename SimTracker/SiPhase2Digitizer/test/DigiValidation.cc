
 // -*- C++ -*-
//
// Package:    DigiValidation
// Class:      DigiValidation
// 
/**\class DigiValidation DigiValidation.cc 

 Description: Test pixel digis. 
 Barrel & Forward digis. Uses root histos.

*/
//
// Author:  Suchandra Dutta
// Created:  July 2013
//
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

// my includes
//#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// data formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

// For the big pixel recongnition
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

// for simulated Tracker hits
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

// For L1
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

// For HLT
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/Common/interface/TriggerNames.h"

// To use root histos
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

// For ROOT
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>
#include <TProfile.h>

//#define HISTOS
//#define L1
//#define HLT

using namespace std;

//
// class declaration
//

class DigiValidation : public edm::EDAnalyzer {

public:

  explicit DigiValidation(const edm::ParameterSet&);
  ~DigiValidation();
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(); 
  
private:
  // ----------member data ---------------------------
  bool PRINT;

  TH1F* nSimTracks_;  

  TH1F* simTrackPt_;  
  TH1F* simTrackPtBar_;  
  TH1F* simTrackPtEC_;  
  TH1F* simTrackEta_;  
  TH1F* simTrackPhi_;  

  TH1F* simTrackPtP_;  
  TH1F* simTrackPtPBar_;  
  TH1F* simTrackPtPEC_;  
  TH1F* simTrackEtaP_;  
  TH1F* simTrackPhiP_;  

  TH1F* simTrackPtS_;
  TH1F* simTrackPtSBar_;  
  TH1F* simTrackPtSEC_;  
  TH1F* simTrackEtaS_;  
  TH1F* simTrackPhiS_;  

  struct DigiHistos {	
    TH1F* NumberOfDigis;
    TH1F* NumberOfDigisP;
    TH1F* NumberOfDigisS;

    TH1F* DigiCharge;
    TH1F* DigiChargeP;
    TH1F* DigiChargeS;

    TH1F* DigiChargeMatched;

    TH1F* NumberOfClusters;
    TH1F* NumberOfClustersP;
    TH1F* NumberOfClustersS;

    TH1F* ClusterCharge;
    TH1F* ClusterChargeP;
    TH1F* ClusterChargeS;

    TH1F* ClusterWidth;
    TH1F* ClusterWidthP;
    TH1F* ClusterWidthS;

    TH1F* TotalNumberOfDigis;
    TH1F* TotalNumberOfClusters;
  
    TH1F* ClusterShape;
    TH1F* ClusterShapeP;
    TH1F* ClusterShapeS;

    TH1F* NumberOfSimHits;
    TH1F* NumberOfMatchedSimHits;    
    TH1F* NumberOfMatchedSimHitsP;
    TH1F* NumberOfMatchedSimHitsS;

    TH1F* DigiEfficiency;
    TH1F* DigiEfficiencyP;
    TH1F* DigiEfficiencyS;

    TH2F* YposVsXpos;
    TH2F* RVsZpos;

    TProfile* ClusterWidthVsSimTrkPt;
    TProfile* ClusterWidthVsSimTrkPtP;
    TProfile* ClusterWidthVsSimTrkPtS;

    TProfile* ClusterWidthVsSimTrkEta;
    TProfile* ClusterWidthVsSimTrkEtaP;
    TProfile* ClusterWidthVsSimTrkEtaS;

    TH1F* matchedSimTrackPt_;  
    TH1F* matchedSimTrackEta_;  
    TH1F* matchedSimTrackPhi_;  
    
    TH1F* matchedSimTrackPtP_;  
    TH1F* matchedSimTrackEtaP_;  
    TH1F* matchedSimTrackPhiP_;  
    
    TH1F* matchedSimTrackPtS_;  
    TH1F* matchedSimTrackEtaS_;  
    TH1F* matchedSimTrackPhiS_;  

    int  totNDigis;
    int  totNClusters; 

    int totSimHits;
    int totSimHitsP;
    int totSimHitsS;

    int totMatchedSimHits; 
    int totMatchedSimHitsP; 
    int totMatchedSimHitsS; 

    std::set<int> simTkIndx;
  };


  struct MyCluster {	
    float charge;
    int   width;
    bool  trkType;      
    float trkPt;
    float trkEta;
    std::vector<float> strip_charges;
  };
  
  //TFile* hFile;
  std::map<unsigned int, DigiHistos> layerHistoMap; 
  
  edm::InputTag src_;  
  edm::InputTag simG4_;  

public:
  void createLayerHistograms(unsigned int iLayer);
  void createHistograms(unsigned int nLayer);
  unsigned int getSimTrackId(edm::Handle<edm::DetSetVector<PixelDigiSimLink> >&, DetId& detId, unsigned int& channel);
  int matchedSimTrack(edm::Handle<edm::SimTrackContainer>& SimTk, unsigned int simTrkId);
  void initializeVariables();
  unsigned int getMaxPosition(std::vector<float>& charge_vec);
  unsigned int getLayerNumber(const TrackerGeometry* tkgeom, unsigned int& detid);
  unsigned int getLayerNumber(unsigned int& detid);
  int isPrimary(const SimTrack& simTrk, edm::Handle<edm::PSimHitContainer>& simHits);
  int isPrimary(const SimTrack& simTrk, const PSimHit& simHit);
  void fillMatchedSimTrackHistos(DigiHistos& digiHistos, const SimTrack& simTk, int ptype, unsigned int layer);
};

//
// constructors and destructor
//
DigiValidation::DigiValidation(const edm::ParameterSet& iConfig) {
  PRINT = iConfig.getUntrackedParameter<bool>("Verbosity",false);
  src_ =  iConfig.getParameter<edm::InputTag>("src");
  simG4_ = iConfig.getParameter<edm::InputTag>("simG4");
  if (PRINT) std::cout << ">>> Construct DigiValidation " << std::endl;
}
DigiValidation::~DigiValidation() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  if (PRINT) std::cout << ">>> Destroy DigiValidation " << std::endl;
}

//
// member functions
//
// ------------ method called at the begining   ------------
void DigiValidation::beginJob() {

   using namespace edm;
   if (PRINT) std::cout << "Initialize DigiValidation " << std::endl;
   createHistograms(15);
  // Create Common Histograms
}

// ------------ method called to producethe data  ------------
void DigiValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //int run       = iEvent.id().run();
  //  int event     = iEvent.id().event();
  //int lumiBlock = iEvent.luminosityBlock();
  //int bx        = iEvent.bunchCrossing();
  //int orbit     = iEvent.orbitNumber();

  // Get digis
  edm::Handle< edm::DetSetVector<PixelDigi> > pixelDigis;
  iEvent.getByLabel(src_, pixelDigis);

  // Get simlink data

  edm::Handle< edm::DetSetVector<PixelDigiSimLink> > pixelSimLinks;
  iEvent.getByLabel(src_,   pixelSimLinks);
  
  // Get event setup (to get global transformation)
  edm::ESHandle<TrackerGeometry> geomHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get( geomHandle );
  const TrackerGeometry*  tkGeom = &(*geomHandle);

  // Get PSimHits
  edm::Handle<edm::PSimHitContainer> simHits;
  iEvent.getByLabel("g4SimHits","TrackerHitsPixelBarrelLowTof" ,simHits);

  edm::Handle<edm::SimTrackContainer> simTracks;
  iEvent.getByLabel("g4SimHits",simTracks);

  // SimVertex
  edm::Handle<edm::SimVertexContainer> simVertices;
  iEvent.getByLabel("g4SimHits", simVertices);
  
  initializeVariables();   

  std::vector<int> processTypes; 
  // Loop over Sim Tracks and Fill relevant histograms
  int nTracks = 0;
  for (edm::SimTrackContainer::const_iterator simTrkItr = simTracks->begin();
                                            simTrkItr != simTracks->end(); ++simTrkItr) {
    int type = isPrimary((*simTrkItr), simHits);
    processTypes.push_back(type);    
    // remove neutrinos
    if (simTrkItr->charge() == 0) continue;
    nTracks++; 
    float simTk_pt =  simTrkItr->momentum().pt();
    float simTk_eta = simTrkItr->momentum().eta();
    float simTk_phi = simTrkItr->momentum().eta();

    
    simTrackPt_->Fill(simTk_pt);
    simTrackEta_->Fill(simTk_eta);
    simTrackPhi_->Fill(simTk_phi);
    if (fabs(simTk_eta) < 1.0) simTrackPtBar_->Fill(simTk_pt);
    else if (fabs(simTk_eta) > 1.6) simTrackPtEC_->Fill(simTk_pt);
 
    if (type == 1) {
      simTrackPtP_->Fill(simTk_pt);
      if (fabs(simTk_eta) < 1.0) simTrackPtPBar_->Fill(simTk_pt);
      else if (fabs(simTk_eta) > 1.6) simTrackPtPEC_->Fill(simTk_pt);
      simTrackEtaP_->Fill(simTk_eta);
      simTrackPhiP_->Fill(simTk_phi);
    } else if (type == 0) {
      simTrackPtS_->Fill(simTk_pt);
      if (fabs(simTk_eta) < 1.0) simTrackPtSBar_->Fill(simTk_pt);
      else if (fabs(simTk_eta) > 1.6) simTrackPtSEC_->Fill(simTk_pt);
      simTrackEtaS_->Fill(simTk_eta);
      simTrackPhiS_->Fill(simTk_phi);
    }
  }
  nSimTracks_->Fill(nTracks++);
 // Loop Over Digis  and Fill Histograms
  edm::DetSetVector<PixelDigi>::const_iterator DSViter;
  for(DSViter = pixelDigis->begin(); DSViter != pixelDigis->end(); DSViter++) {
    unsigned int rawid = DSViter->id; 
    DetId detId(rawid);
    unsigned int layer = getLayerNumber(rawid);
    std::map<unsigned int, DigiHistos>::iterator iPos = layerHistoMap.find(layer);
    if (iPos == layerHistoMap.end()) {
      createLayerHistograms(layer);
      iPos = layerHistoMap.find(layer);
    }
    const GeomDetUnit* geomDetUnit = tkGeom->idToDetUnit(detId);
    //    const PixelGeomDetUnit* pixdet = (PixelGeomDetUnit*) geomDetUnit;
    edm::DetSet<PixelDigi>::const_iterator di;
    int col_last  = -1;
    int row_last  = -1;
    int nDigiP = 0;
    int nDigiS = 0;
    vector<MyCluster> cluster_vec;
    MyCluster cluster;
    cluster.charge = 0.0;
    cluster.width = 0;
    cluster.trkType = false;
    cluster.trkPt   = -999.0;
    cluster.trkEta  = -999.0;
    cluster.strip_charges.clear();
    for(di = DSViter->data.begin(); di != DSViter->data.end(); di++) {
      int adc = di->adc();    // charge, modifued to unsiged short 
      int col = di->column(); // column 
      int row = di->row();    // row

      unsigned int channel = PixelChannelIdentifier::pixelToChannel(row,col);
      unsigned int simTkId = getSimTrackId(pixelSimLinks, detId, channel);
      MeasurementPoint mp(row+0.5, col+0.5 );

      if (geomDetUnit) {
	GlobalPoint pdPos = geomDetUnit->surface().toGlobal( geomDetUnit->topology().localPosition( mp ) ) ;
	iPos->second.YposVsXpos->Fill(pdPos.y(), pdPos.x());
	iPos->second.RVsZpos->Fill(pdPos.z(), pdPos.perp());
      }
      int iSimTrk = matchedSimTrack(simTracks, simTkId);


      int primaryTrk = -1; 
      if (iSimTrk != -1) {
	primaryTrk = processTypes[iSimTrk]; 
	iPos->second.simTkIndx.insert(iSimTrk);
      }
      iPos->second.DigiCharge->Fill(adc); 

      if (primaryTrk == 1) {
	iPos->second.DigiChargeP->Fill(adc); 

        nDigiP++;
      } else if (primaryTrk == 0){
	iPos->second.DigiChargeS->Fill(adc);
	nDigiS++;
      }
      if (row_last == -1 ) {
        cluster.charge = adc;
        cluster.width  = 1;
        cluster.trkType = primaryTrk;  
	cluster.trkPt = (*simTracks)[iSimTrk].momentum().pt();
	cluster.trkEta = (*simTracks)[iSimTrk].momentum().eta();
        cluster.strip_charges.clear();
        cluster.strip_charges.push_back(adc);
      } else {

	if (abs(row - row_last) == 1 && col == col_last) {
	  cluster.charge += adc;
	  cluster.width++;
	  cluster.strip_charges.push_back(adc);
	} else {
	  cluster_vec.push_back(cluster);
	  cluster.charge = adc;
	  cluster.width  = 1;
	  cluster.trkType = primaryTrk;  
          cluster.trkPt = (*simTracks)[iSimTrk].momentum().pt();
	  cluster.trkEta = (*simTracks)[iSimTrk].momentum().eta();
	  cluster.strip_charges.clear();
	  cluster.strip_charges.push_back(adc);
	}
      }
      if (PRINT) std::cout << ">>> PType " << primaryTrk 
                           << " SimTkPt " << (*simTracks)[iSimTrk].momentum().pt() 
                           << " column " << col 
                           << " row " << row 
                           << " column_last " << col_last 
                           << " row_last " << row_last 
                           << " nDigi " << nDigiS 
                           << " " << nDigiP  
                           << " nCluster " << cluster_vec.size() 
                           << " Cluster Charge " << cluster.charge 
                           << " Cluster Width " << cluster.width << std::endl;
      col_last = col;
      row_last = row;
    }
    cluster_vec.push_back(cluster);
    if (PRINT) std::cout << ">>> detid: " << DSViter->id << " Cluster size " << cluster_vec.size() << std::endl; 
    iPos->second.NumberOfDigisP->Fill(nDigiP);
    iPos->second.NumberOfDigisS->Fill(nDigiS);
    iPos->second.NumberOfDigis->Fill(nDigiP+nDigiS);
    iPos->second.totNDigis += nDigiP + nDigiS;

    int nClusterP = 0;
    int nClusterS = 0;
  
    for (vector<MyCluster>::iterator ic = cluster_vec.begin(); ic != cluster_vec.end(); ++ic) {
      float cl_charge = ic->charge;
      int  cl_width = ic->width;
      int  cl_type = ic->trkType; 
      float trk_pt = ic->trkPt;
      float trk_eta = ic->trkEta;
      std::vector<float> str_charges = ic->strip_charges;
      unsigned int max_pos = getMaxPosition(str_charges);
      if (max_pos != 999) {
        for (unsigned int ival = 0; ival < str_charges.size(); ++ival) {
          int pos = ival - max_pos; 
          iPos->second.ClusterShape->Fill(pos, str_charges[ival]); 
          if (cl_type == 1) iPos->second.ClusterShapeP->Fill(pos, str_charges[ival]); 
          else if (cl_type == 0) iPos->second.ClusterShapeS->Fill(pos, str_charges[ival]); 
        }
      }
      iPos->second.ClusterCharge->Fill(cl_charge);           
      iPos->second.ClusterWidth->Fill(cl_width);           
      iPos->second.ClusterWidthVsSimTrkPt->Fill(trk_pt, cl_width);            
      iPos->second.ClusterWidthVsSimTrkEta->Fill(trk_eta, cl_width);            
      if (cl_type == 1) {
	iPos->second.ClusterChargeP->Fill(cl_charge);           
	iPos->second.ClusterWidthP->Fill(cl_width); 
        iPos->second.ClusterWidthVsSimTrkPtP->Fill(trk_pt, cl_width);
	iPos->second.ClusterWidthVsSimTrkEtaP->Fill(trk_eta, cl_width);                        
	nClusterP++; 
      } else if (cl_type == 0) {
	iPos->second.ClusterChargeS->Fill(cl_charge);           
	iPos->second.ClusterWidthS->Fill(cl_width);           
        iPos->second.ClusterWidthVsSimTrkPtS->Fill(trk_pt, cl_width);            
        iPos->second.ClusterWidthVsSimTrkEtaS->Fill(trk_eta, cl_width);            
	nClusterS++; 
      }
    }
    iPos->second.NumberOfClustersP->Fill(nClusterP);
    iPos->second.NumberOfClustersS->Fill(nClusterS);
    iPos->second.NumberOfClusters->Fill(nClusterP+nClusterS);
    iPos->second.totNClusters += nClusterP + nClusterS;
  }
  // Fill Layer Level Histograms
  for (std::map<unsigned int, DigiHistos>::iterator iPos  = layerHistoMap.begin(); 
                                                    iPos != layerHistoMap.end(); iPos++) {
    DigiHistos local_histos = iPos->second;
    local_histos.TotalNumberOfClusters->Fill(local_histos.totNClusters);
    local_histos.TotalNumberOfDigis->Fill(local_histos.totNDigis);
    local_histos.NumberOfSimHits->Fill(local_histos.totSimHits);
    local_histos.NumberOfMatchedSimHits->Fill(local_histos.totMatchedSimHits);
    local_histos.NumberOfMatchedSimHitsP->Fill(local_histos.totMatchedSimHitsP);
    local_histos.NumberOfMatchedSimHitsS->Fill(local_histos.totMatchedSimHitsS);

    if (local_histos.totSimHits) {
      float eff;
      eff  = local_histos.totMatchedSimHits*1.0/local_histos.totSimHits; 
      local_histos.DigiEfficiency->Fill(eff);
      eff  = local_histos.totMatchedSimHitsP*1.0/local_histos.totSimHitsP; 
      local_histos.DigiEfficiencyP->Fill(eff);
      eff  = local_histos.totMatchedSimHitsS*1.0/local_histos.totSimHitsS; 
      local_histos.DigiEfficiencyS->Fill(eff);
    }

    for (std::set<int>::iterator ii = local_histos.simTkIndx.begin(); ii != local_histos.simTkIndx.end(); ii++) {
      unsigned int index = (*ii);
      int pid = processTypes[index];  
      fillMatchedSimTrackHistos(local_histos,(*simTracks.product())[index], pid, iPos->first);
    }
  }
}
// ------------ method called to at the end of the job  ------------
void DigiValidation::endJob(){
  //hFile->Write();
  //hFile->Close();
}
// ------------ method called to create histograms for a specific layer  ------------
void DigiValidation::createLayerHistograms(unsigned int ival) {
  std::ostringstream fname1, fname2;
  
  edm::Service<TFileService> fs;
  fs->file().cd("/");
  
  std::string tag;
  unsigned int id; 
  if (ival < 100) { 
    id = ival;
    fname1 << "Barrel";
    fname2 << "Layer_" << id;    
    tag = "_layer_";
   } else {
    int side = ival/100;
    id = ival - side*100; 
    std::cout << " Creating histograms for Disc " << id << " with " << ival << std::endl; 
    fname1 << "EndCap_Side_" << side; 
    fname2 << "Disc_" << id;       
    tag = "_disc_";
  }
  TFileDirectory td1 = fs->mkdir(fname1.str().c_str());
  TFileDirectory td = td1.mkdir(fname2.str().c_str());
     
  DigiHistos local_histos;
  std::ostringstream htit1;
  htit1 << "NumberOfDigis" << tag.c_str() <<  id;   
  local_histos.NumberOfDigis = td.make<TH1F>(htit1.str().c_str(), htit1.str().c_str(), 51, -0.5, 50.5);
  htit1.str("");
  htit1 << "NumberOfDigisP" << tag.c_str() <<  id;   
  local_histos.NumberOfDigisP = td.make<TH1F>(htit1.str().c_str(), htit1.str().c_str(), 51, -0.5, 50.5);
  htit1.str("");
  htit1 << "NumberOfDigisS" << tag.c_str() <<  id;   
  local_histos.NumberOfDigisS = td.make<TH1F>(htit1.str().c_str(), htit1.str().c_str(), 51, -0.5, 50.5);

  
  std::ostringstream htit2;
  htit2 << "DigiCharge" << tag.c_str() <<  id;   
  local_histos.DigiCharge = td.make<TH1F>(htit2.str().c_str(), htit2.str().c_str(), 261, -0.5, 260.5);
  htit2.str("");
  htit2 << "DigiChargeP" << tag.c_str() <<  id;   
  local_histos.DigiChargeP = td.make<TH1F>(htit2.str().c_str(), htit2.str().c_str(), 261, -0.5, 260.5);
  htit2.str("");
  htit2 << "DigiChargeS" << tag.c_str() <<  id;   
  local_histos.DigiChargeS = td.make<TH1F>(htit2.str().c_str(), htit2.str().c_str(), 261, -0.5, 260.5);
  
  std::ostringstream htit3;
  htit3 << "NumberOfClusters" << tag.c_str() <<  id;   
  local_histos.NumberOfClusters = td.make<TH1F>(htit3.str().c_str(), htit3.str().c_str(), 51, -0.5, 50.5);
  htit3.str("");  
  htit3 << "NumberOfClustersP" << tag.c_str() <<  id;   
  local_histos.NumberOfClustersP = td.make<TH1F>(htit3.str().c_str(), htit3.str().c_str(), 51, -0.5, 50.5);
  htit3.str("");  
  htit3 << "NumberOfClustersS" << tag.c_str() <<  id;   
  local_histos.NumberOfClustersS = td.make<TH1F>(htit3.str().c_str(), htit3.str().c_str(), 51, -0.5, 50.5);

  std::ostringstream htit4;
  htit4 << "ClusterCharge" << tag.c_str() <<  id;   
  local_histos.ClusterCharge = td.make<TH1F>(htit4.str().c_str(), htit4.str().c_str(), 1041, -0.5, 1040.5);
  htit4.str("");
  htit4 << "ClusterChargeP" << tag.c_str() <<  id;   
  local_histos.ClusterChargeP = td.make<TH1F>(htit4.str().c_str(), htit4.str().c_str(), 1041, -0.5,1040.5);
  htit4.str("");
  htit4 << "ClusterChargeS" << tag.c_str() <<  id;   
  local_histos.ClusterChargeS = td.make<TH1F>(htit4.str().c_str(), htit4.str().c_str(), 1041, -0.5, 1040.5);
  
  std::ostringstream htit5;
  htit5 << "ClusterWidth" << tag.c_str() <<  id;   
  local_histos.ClusterWidth = td.make<TH1F>(htit5.str().c_str(), htit5.str().c_str(), 16, -0.5, 15.5);
  htit5.str("");
  htit5 << "ClusterWidthP" << tag.c_str() <<  id;   
  local_histos.ClusterWidthP = td.make<TH1F>(htit5.str().c_str(), htit5.str().c_str(), 16, -0.5, 15.5);
  htit5.str("");
  htit5 << "ClusterWidthS" << tag.c_str() <<  id;   
  local_histos.ClusterWidthS = td.make<TH1F>(htit5.str().c_str(), htit5.str().c_str(), 16, -0.5, 15.5);
  
  std::ostringstream htit6;
  htit6 << "TotalNumberOfDigis" << tag.c_str() <<  id;   
  local_histos.TotalNumberOfDigis = td.make<TH1F>(htit6.str().c_str(), htit6.str().c_str(), 100, -0.5, 100.5);
  
  std::ostringstream htit7;
  htit7 << "TotalNumberOfClusters" << tag.c_str() <<  id;   
  local_histos.TotalNumberOfClusters = td.make<TH1F>(htit7.str().c_str(), htit7.str().c_str(), 100, -0.5, 100.5);

  std::ostringstream htit8;
  htit8 << "ClusterShape" << tag.c_str() <<  id;   
  local_histos.ClusterShape = td.make<TH1F>(htit8.str().c_str(), htit8.str().c_str(), 21, -20.5, 20.5);
  htit8.str("");
  htit8<< "ClusterShapeP" << tag.c_str() <<  id;   
  local_histos.ClusterShapeP = td.make<TH1F>(htit8.str().c_str(), htit8.str().c_str(), 21, -20.5, 20.5);
  htit8.str("");
  htit8 << "ClusterShapeS" << tag.c_str() <<  id;   
  local_histos.ClusterShapeS = td.make<TH1F>(htit8.str().c_str(), htit8.str().c_str(), 21, -20.5, 20.5);

  std::ostringstream htit9;
  htit9 << "NumberOfSimHits" << tag.c_str() <<  id;   
  local_histos.NumberOfSimHits = td.make<TH1F>(htit9.str().c_str(), htit9.str().c_str(), 201, -0.5, 200.5);
  htit9.str("");
  htit9 << "NumberOfMatchedSimHits" << tag.c_str() <<  id;   
  local_histos.NumberOfMatchedSimHits = td.make<TH1F>(htit9.str().c_str(), htit9.str().c_str(), 201, -0.5, 200.5);
  htit9.str("");
  htit9 << "NumberOfMatchedSimHitsP" << tag.c_str() <<  id;   
  local_histos.NumberOfMatchedSimHitsP = td.make<TH1F>(htit9.str().c_str(), htit9.str().c_str(), 201, -0.5, 200.5);
  htit9.str("");
  htit9 << "NumberOfMatchedSimHitsS" << tag.c_str() <<  id;   
  local_histos.NumberOfMatchedSimHitsS = td.make<TH1F>(htit9.str().c_str(), htit9.str().c_str(), 201, -0.5, 200.5);

  std::ostringstream htit10;
  htit10 << "DigiEfficiency" << tag.c_str() <<  id;   
  local_histos.DigiEfficiency = td.make<TH1F>(htit10.str().c_str(), htit10.str().c_str(), 55, -0.05, 1.05);
  htit10.str("");
  htit10 << "DigiEfficiencyP" << tag.c_str() <<  id;   
  local_histos.DigiEfficiencyP = td.make<TH1F>(htit10.str().c_str(), htit10.str().c_str(), 55, -0.05, 1.05);
  htit10.str("");
  htit10 << "DigiEfficiencyS" << tag.c_str() <<  id;   
  local_histos.DigiEfficiencyS = td.make<TH1F>(htit10.str().c_str(), htit10.str().c_str(), 55, -0.05, 1.05);
  
  std::ostringstream htit11;
  htit11 << "YposVsXpos" << tag.c_str() <<  id;   
  local_histos.YposVsXpos = td.make<TH2F>(htit11.str().c_str(), htit11.str().c_str(), 240, -120.0, 120.0, 240, -120.0, 120.0);

  std::ostringstream htit12;
  htit12 << "RVsZpos" << tag.c_str() <<  id;   
  local_histos.RVsZpos = td.make<TH2F>(htit12.str().c_str(), htit12.str().c_str(), 600, -300.0, 300.0, 120, 0.0, 120.0);

  std::ostringstream htit13;
  htit13 << "DigiChargeMatched" << tag.c_str() <<  id;   
  local_histos.DigiChargeMatched = td.make<TH1F>(htit13.str().c_str(), htit13.str().c_str(), 261, -0.5, 260.5);

  std::ostringstream htit14;
  htit14 << "ClusterWidthVsSimTrkPt" << tag.c_str() <<  id;   
  local_histos.ClusterWidthVsSimTrkPt = td.make<TProfile>(htit14.str().c_str(),htit14.str().c_str(),56, -0.5, 55.5,-0.5,15.5);
  htit14.str("");
  htit14 << "ClusterWidthVsSimTrkPtP" << tag.c_str() <<  id;   
  local_histos.ClusterWidthVsSimTrkPtP = td.make<TProfile>(htit14.str().c_str(),htit14.str().c_str(),56, -0.5, 55.5,-0.5,15.5);
  htit14.str("");
  htit14 << "ClusterWidthVsSimTkrPtS" << tag.c_str() <<  id;   
  local_histos.ClusterWidthVsSimTrkPtS = td.make<TProfile>(htit14.str().c_str(),htit14.str().c_str(),56, -0.5, 55.5,-0.5,15.5);

  std::ostringstream htit15;
  htit15 << "ClusterWidthVsSimTrkEta" << tag.c_str() <<  id;   
  local_histos.ClusterWidthVsSimTrkEta = td.make<TProfile>(htit15.str().c_str(),htit15.str().c_str(),50, -2.5, 2.5,-0.5,15.5);
  htit15.str("");
  htit15 << "ClusterWidthVsSimTrkEtaP" << tag.c_str() <<  id;   
  local_histos.ClusterWidthVsSimTrkEtaP = td.make<TProfile>(htit15.str().c_str(),htit15.str().c_str(),50, -2.5, 2.5,-0.5,15.5);
  htit15.str("");
  htit15 << "ClusterWidthVsSimTkrEtaS" << tag.c_str() <<  id;   
  local_histos.ClusterWidthVsSimTrkEtaS = td.make<TProfile>(htit15.str().c_str(),htit15.str().c_str(),50, -2.5, 2.5,-0.5,15.5);

  std::ostringstream htit16;
  htit16 << "MatchedSimTrackPt" << tag.c_str() <<  id;   
  local_histos.matchedSimTrackPt_  = td.make<TH1F>(htit16.str().c_str(),htit16.str().c_str(),101,-0.5,100.5);
  htit16.str("");
  htit16 << "MatchedSimTrackPtP" << tag.c_str() <<  id;   
  local_histos.matchedSimTrackPtP_  = td.make<TH1F>(htit16.str().c_str(),htit16.str().c_str(),101,-0.5,100.5);
  htit16.str("");
  htit16 << "MatchedSimTrackPtS" << tag.c_str() <<  id;   
  local_histos.matchedSimTrackPtS_  = td.make<TH1F>(htit16.str().c_str(),htit16.str().c_str(),101,-0.5,100.5);

  std::ostringstream htit17;
  htit17 << "MatchedSimTrackEta" << tag.c_str() <<  id;   
  local_histos.matchedSimTrackEta_  = td.make<TH1F>(htit17.str().c_str(),  htit17.str().c_str(), 50, -2.5, 2.5);
  htit17.str("");
  htit17 << "MatchedSimTrackEtaP" << tag.c_str() <<  id;   
  local_histos.matchedSimTrackEtaP_  = td.make<TH1F>(htit17.str().c_str(),  htit17.str().c_str(), 50, -2.5, 2.5);
  htit17.str("");
  htit17 << "MatchedSimTrackEtaS" << tag.c_str() <<  id;   
  local_histos.matchedSimTrackEtaS_  = td.make<TH1F>(htit17.str().c_str(),  htit17.str().c_str(), 50, -2.5, 2.5);

  std::ostringstream htit18;
  htit18 << "MatchedSimTrackPhi" << tag.c_str() <<  id;   
  local_histos.matchedSimTrackPhi_  = td.make<TH1F>(htit18.str().c_str(),  htit18.str().c_str(), 160, -3.2, 3.2);
  htit18.str("");
  htit18 << "MatchedSimTrackPhiP" << tag.c_str() <<  id;   
  local_histos.matchedSimTrackPhiP_  = td.make<TH1F>(htit18.str().c_str(),  htit18.str().c_str(), 160, -3.2, 3.2);
  htit18.str("");
  htit18 << "MatchedSimTrackPhiS" << tag.c_str() <<  id;   
  local_histos.matchedSimTrackPhiS_  = td.make<TH1F>(htit18.str().c_str(),  htit18.str().c_str(), 160, -3.2, 3.2);

  layerHistoMap.insert( std::make_pair(ival, local_histos));

  fs->file().cd("/");
 
  layerHistoMap[ival].totNDigis    = 0;
  layerHistoMap[ival].totNClusters = 0;
  layerHistoMap[ival].totSimHits   = 0;
  layerHistoMap[ival].totSimHitsP  = 0;
  layerHistoMap[ival].totSimHitsS  = 0;
  layerHistoMap[ival].totMatchedSimHits  = 0;
  layerHistoMap[ival].totMatchedSimHitsP = 0;
  layerHistoMap[ival].totMatchedSimHitsS = 0;
  
  layerHistoMap[ival].simTkIndx.clear();
}

// ------------ method called to create histograms for all layers  ------------
void DigiValidation::createHistograms(unsigned int nLayer) {

  // NEW way to use root (from 2.0.0?)
  edm::Service<TFileService> fs;
  fs->file().cd("/");
  TFileDirectory td = fs->mkdir("Common");

  nSimTracks_  = td.make<TH1F>("nSimTracks", "Number of Sim Tracks" , 201, -0.5, 200.5);
  simTrackPt_  = td.make<TH1F>("SimTrackPt", "Pt of Sim Tracks", 101, -0.5, 100.5);
  simTrackPtBar_  = td.make<TH1F>("SimTrackPtBar", "Pt of Sim Tracks( eta < 1.0)", 101, -0.5, 100.5);
  simTrackPtEC_  = td.make<TH1F>("SimTrackPtEC", "Pt of Sim Tracks( eta > 1.6)", 101, -0.5, 100.5);
  simTrackEta_ =  td.make<TH1F>("SimTrackEta", "Eta of Sim Tracks", 50, -2.5, 2.5);
  simTrackPhi_ =  td.make<TH1F>("SimTrackPhi", "Phi of Sim Tracks", 160, -3.2, 3.2);

  simTrackPtP_  = td.make<TH1F>("SimTrackPtP", "Pt of Primary Sim Tracks", 101, -0.5, 100.5);
  simTrackPtPBar_  = td.make<TH1F>("SimTrackPtPBar", "Pt of Primary Sim Tracks( eta < 1.0)", 101, -0.5, 100.5);
  simTrackPtPEC_  = td.make<TH1F>("SimTrackPtPEC", "Pt of Primary Sim Tracks( eta > 1.6)", 101, -0.5, 100.5);
  simTrackEtaP_ =  td.make<TH1F>("SimTrackEtaP", "Eta of Primary Sim Tracks", 50, -2.5, 2.5);
  simTrackPhiP_ =  td.make<TH1F>("SimTrackPhiP", "Phi of Primary Sim Tracks", 160, -3.2, 3.2);

  simTrackPtS_  = td.make<TH1F>("SimTrackPtS", "Pt of Secondary Sim Tracks", 101, -0.5, 100.5);
  simTrackPtSBar_  = td.make<TH1F>("SimTrackPtSBar", "Pt of Secondary Sim Tracks( eta < 1.0)", 101, -0.5, 100.5);
  simTrackPtSEC_  = td.make<TH1F>("SimTrackPtSEC", "Pt of Secondary Sim Tracks( eta > 1.6)", 101, -0.5, 100.5);
  simTrackEtaS_ =  td.make<TH1F>("SimTrackEtaS", "Eta of Secondary Sim Tracks", 50, -2.5, 2.5);
  simTrackPhiS_ =  td.make<TH1F>("SimTrackPhiS", "Phi of Secondary Sim Tracks", 160, -3.2, 3.2);
}
// ------------ method called to create histograms for all layers  ------------
unsigned int DigiValidation::getSimTrackId(edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& pixelSimLinks, DetId& detId, unsigned int& channel) {

  edm::DetSetVector<PixelDigiSimLink>::const_iterator 
    isearch = pixelSimLinks->find(detId);

  unsigned int simTrkId(0);
  if (isearch == pixelSimLinks->end()) return simTrkId;

  edm::DetSet<PixelDigiSimLink> link_detset = (*pixelSimLinks)[detId];
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

int DigiValidation::matchedSimTrack(edm::Handle<edm::SimTrackContainer>& SimTk, unsigned int simTrkId) {

  edm::SimTrackContainer sim_tracks = (*SimTk.product());
  for(unsigned int it = 0; it < sim_tracks.size(); it++) {
    if (sim_tracks[it].trackId() == simTrkId) {
      return it;
    }
  }
  return -1;
}
//
// Initialize Digi/Cluster Counters 
//
void DigiValidation::initializeVariables() {
  for (std::map<unsigned int, DigiHistos>::iterator iPos = layerHistoMap.begin(); iPos != layerHistoMap.end(); iPos++) {
    iPos->second.totNDigis    = 0;
    iPos->second.totNClusters = 0;
    iPos->second.totSimHits    = 0;
    iPos->second.totSimHitsP   = 0;
    iPos->second.totSimHitsS   = 0;
    iPos->second.totMatchedSimHits = 0;
    iPos->second.totMatchedSimHitsP = 0;
    iPos->second.totMatchedSimHitsS = 0;
    iPos->second.simTkIndx.clear(); 
  }
}
//
// -- Get Layer Number
//
unsigned int DigiValidation::getLayerNumber(const TrackerGeometry* tkgeom,unsigned int& detid) {
  unsigned int layer = 999;
  DetId theDetId(detid);
  if (theDetId.subdetId() != 1) 
    std::cout << ">>> Method1 : Det id " << theDetId.det() << " Subdet Id " << theDetId.subdetId() << std::endl;  
  const PixelGeomDetUnit * theGeomDet = 
      dynamic_cast<const PixelGeomDetUnit*> ( tkgeom->idToDet(theDetId) );

  const GeomDetUnit* it = tkgeom->idToDetUnit(DetId(theDetId));
  if (!it) std::cout << ">>> rawdetid " << detid  
                     << " GeomDetUnit " << it 
                     << " PixelGeomDetUnit " << theGeomDet 
                     << " DetId " << theDetId.det() 
                     << " Subdet Id " << theDetId.subdetId() 
                     << std::endl;

  if (it && it->type().isTracker()) {
    if (it->type().isBarrel()) {
      PXBDetId pb_detId = PXBDetId(detid);
      layer = pb_detId.layer();
    } else if (it->type().isEndcap()) {
      PXFDetId pf_detId = PXFDetId(detid);
      layer = 100*pf_detId.side() + pf_detId.disk();
    }
  }
  return layer;
}
//
// -- Get Layer Number
//
unsigned int DigiValidation::getLayerNumber(unsigned int& detid) {
  unsigned int layer = 999;
  DetId theDetId(detid);
  if (theDetId.det() == DetId::Tracker) {
    if (theDetId.subdetId() == PixelSubdetector::PixelBarrel) {
      PXBDetId pb_detId = PXBDetId(detid);
      layer = pb_detId.layer();
    } else if (theDetId.subdetId() == PixelSubdetector::PixelEndcap) {
      PXFDetId pf_detId = PXFDetId(detid);
      layer = 100*pf_detId.side() + pf_detId.disk();
    } else {
      std::cout << ">>> Invalid subdetId() = " << theDetId.subdetId() << std::endl;
    }
  }
  return layer;
}
//
// -- Get Maximun position of a vector
//
unsigned int DigiValidation::getMaxPosition(std::vector<float>& charge_vec) {
  unsigned int ipos = 999;
  float max_val = 0.0;
  for (unsigned int ival = 0; ival < charge_vec.size(); ++ival) {
    if (charge_vec[ival] > max_val) {
      max_val = charge_vec[ival];
      ipos = ival;
    }
  }
  return ipos;
}
//
//  -- Check if the SimTrack is _Primary or not 
//
int DigiValidation::isPrimary(const SimTrack& simTrk, edm::Handle<edm::PSimHitContainer>& simHits) {
  int result = -1;
  unsigned int trkId = simTrk.trackId();
  if (trkId > 0) {
    int vtxIndx = simTrk.vertIndex();
    for (edm::PSimHitContainer::const_iterator iHit = simHits->begin(); iHit != simHits->end(); ++iHit) {
      if (trkId == iHit->trackId()) {
	int ptype = iHit->processType();

 
	if ( (vtxIndx == 0 ) && (ptype == 2 || ptype == 7 || ptype == 9 || ptype == 11 || ptype == 15)) result = 1;
        else result = 0; 
	break;
      }
    }
  }
  return result;
}
//
//  -- Check if the SimTrack is _Primary or not 
//
int DigiValidation::isPrimary(const SimTrack& simTrk, const PSimHit& simHit) {
  int result = -1;
  unsigned int trkId = simTrk.trackId();
  if (trkId > 0) {
    int vtxIndx = simTrk.vertIndex();
    int ptype = simHit.processType();
    if ( (vtxIndx == 0 ) && (ptype == 2 || ptype == 7 || ptype == 9 || ptype == 11 || ptype == 15)) result = 1;
    else result = 0;
  }
  return result;
}
void DigiValidation::fillMatchedSimTrackHistos(DigiHistos& digiHistos, const SimTrack& simTk, int ptype, unsigned int layer){ 
  
  float pt =  simTk.momentum().pt();
  float eta = simTk.momentum().eta();
  float phi = simTk.momentum().phi();

  if (layer < 100 && fabs(eta) < 1.0)  {
    digiHistos.matchedSimTrackPt_->Fill(pt); 
    if (ptype == 1)  digiHistos.matchedSimTrackPtP_->Fill(pt);
    else if (ptype == 0)  digiHistos.matchedSimTrackPtS_->Fill(pt);
  } else if (layer > 100 && fabs(eta) > 1.6) {
    digiHistos.matchedSimTrackPt_->Fill(pt); 
    if (ptype == 1)  digiHistos.matchedSimTrackPtP_->Fill(pt);
    else if (ptype == 0)  digiHistos.matchedSimTrackPtS_->Fill(pt);
  } 
  if (ptype == 1) {
    digiHistos.matchedSimTrackEtaP_->Fill(eta);  
    digiHistos.matchedSimTrackPhiP_->Fill(phi);  
  } else if (ptype == 0){
    digiHistos.matchedSimTrackEtaS_->Fill(eta);  
    digiHistos.matchedSimTrackPhiS_->Fill(phi);  
  }
  digiHistos.matchedSimTrackEta_->Fill(eta);  
  digiHistos.matchedSimTrackPhi_->Fill(phi);  
}
//define this as a plug-in
DEFINE_FWK_MODULE(DigiValidation);
