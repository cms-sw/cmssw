// -*- C++ -*-
//
// Package:    HSCP
// Class:      HSCPValidator
// 
/**\class HSCPValidator HSCPValidator.cc HSCPValidation/HSCPValidator/src/HSCPValidator.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Seth Cooper,27 1-024,+41227672342,
//         Created:  Wed Apr 14 14:27:52 CEST 2010
// $Id: HSCPValidator.cc,v 1.9 2011/10/11 21:14:33 jiechen Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <string>
#include <map>

// user include files
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h>
#include <DataFormats/GeometryVector/interface/LocalPoint.h>
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h>
#include <DataFormats/GeometryVector/interface/LocalPoint.h>
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "FastSimulation/Tracking/test/FastTrackAnalyzer.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SUSYBSMAnalysis/HSCP/interface/HSCPValidator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

#include "TH1.h"
#include "TGraph.h"
#include "TCanvas.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
edm::Service<TFileService> fileService;

//
// constructors and destructor
//
HSCPValidator::HSCPValidator(const edm::ParameterSet& iConfig) :
  doGenPlots_ (iConfig.getParameter<bool>("MakeGenPlots")),
  doHLTPlots_ (iConfig.getParameter<bool>("MakeHLTPlots")),
  doSimTrackPlots_ (iConfig.getParameter<bool>("MakeSimTrackPlots")),
  doSimDigiPlots_ (iConfig.getParameter<bool>("MakeSimDigiPlots")),
  doRecoPlots_ (iConfig.getParameter<bool>("MakeRecoPlots")),
  label_ (iConfig.getParameter<edm::InputTag>("generatorLabel")),
  particleIds_ (iConfig.getParameter< std::vector<int> >("particleIds")),
  particleStatus_ (iConfig.getUntrackedParameter<int>("particleStatus",1)),
  ebSimHitTag_ (iConfig.getParameter<edm::InputTag>("EBSimHitCollection")),
  eeSimHitTag_ (iConfig.getParameter<edm::InputTag>("EESimHitCollection")),
  simTrackTag_ (iConfig.getParameter<edm::InputTag>("SimTrackCollection")),
  EBDigiCollection_ (iConfig.getParameter<edm::InputTag>("EBDigiCollection")),
  EEDigiCollection_ (iConfig.getParameter<edm::InputTag>("EEDigiCollection")),
  RPCRecHitTag_ (iConfig.getParameter<edm::InputTag>("RPCRecHitTag"))
{
  //now do what ever initialization is needed
  // GEN
  particleEtaHist_ = fileService->make<TH1F>("particleEta","Eta of gen particle",100,-5,5);
  particlePhiHist_ = fileService->make<TH1F>("particlePhi","Phi of gen particle",180,-3.15,3.15);
  particlePHist_ = fileService->make<TH1F>("particleP","Momentum of gen particle",500,0,2000);
  particlePtHist_ = fileService->make<TH1F>("particlePt","P_{T} of gen particle",500,0,2000);
  particleMassHist_ = fileService->make<TH1F>("particleMass","Mass of gen particle",1000,0,2000);
  particleStatusHist_ = fileService->make<TH1F>("particleStatus","Status of gen particle",10,0,10);
  particleBetaHist_ = fileService->make<TH1F>("particleBeta","Beta of gen particle",100,0,1);
  particleBetaInverseHist_ = fileService->make<TH1F>("particleBetaInverse","1/#beta of gen particle",100,0,5);
  
  h_genhscp_met = fileService->make<TH1F>( "hscp_met"  , "missing E_{T} hscp" , 100, 0., 1500. );
  h_genhscp_met_nohscp = fileService->make<TH1F>( "hscp_met_nohscp"  , "missing E_{T} w/o hscp" , 100, 0., 1500. );
  h_genhscp_scaloret =  fileService->make<TH1F>( "hscp_scaloret"  , "scalor E_{T} sum" , 100, 0., 1500. );
  h_genhscp_scaloret_nohscp =  fileService->make<TH1F>( "hscp_scaloret_nohscp"  , "scalor E_{T} sum w/o hscp" , 100, 0., 1500. );




  //SIM track Info
  simTrackParticleEtaHist_ = fileService->make<TH1F>("simTrackParticleEta","Eta of simTrackParticle",100,-5,5);
  simTrackParticlePhiHist_ = fileService->make<TH1F>("simTrackParticlePhi","Phi of simTrackParticle",180,-3.15,3.15);
  simTrackParticlePHist_ = fileService->make<TH1F>("simTrackParticleP","Momentum of simTrackParticle",500,0,2000);
  simTrackParticlePtHist_ = fileService->make<TH1F>("simTrackParticlePt","P_{T} of simTrackParticle",500,0,2000);
  simTrackParticleBetaHist_ = fileService->make<TH1F>("simTrackParticleBeta","Beta of simTrackParticle",100,0,1);
//reco track Info

  RecoHSCPPtVsGenPt= fileService->make<TH2F>("Recovsgenpt","RecovsGen",100,0,1000,100,0,1000);
  dedxVsp = fileService->make<TH2F>("dedxvsp","dedxvsp",100,0,1000,100,0,10);
//HLT Info
  hltmet = fileService->make<TH1F>("HLT_MET","MET",3,-1,2);
  hltjet = fileService->make<TH1F>("HLT_JET","JET",3,-1,2);
  hltmu = fileService->make<TH1F>("HLT_Mu","Mu",3,-1,2);


  // SIM-DIGI: ECAL
  simHitsEcalEnergyHistEB_ = fileService->make<TH1F>("ecalEnergyOfSimHitsEB","HSCP SimTrack-matching SimHit energy EB [GeV]",125,-1,4);
  simHitsEcalEnergyHistEE_ = fileService->make<TH1F>("ecalEnergyOfSimHitsEE","HSCP SimTrack-matching SimHit energy EE [GeV]",125,-1,4);
  simHitsEcalTimeHistEB_ = fileService->make<TH1F>("ecalTimingOfSimHitsEB","HSCP SimTrack-matching SimHit time EB [ns]",115,-15,100);
  simHitsEcalTimeHistEE_ = fileService->make<TH1F>("ecalTimingOfSimHitsEE","HSCP SimTrack-matching SimHit time EE [ns]",115,-15,100);
  simHitsEcalNumHistEB_ = fileService->make<TH1F>("ecalNumberOfSimHitsEB","Number of HSCP SimTrack-matching EB sim hits in event",100,0,200);
  simHitsEcalNumHistEE_ = fileService->make<TH1F>("ecalNumberOfSimHitsEE","Number of HSCP SimTrack-matching EE sim hits in event",100,0,200);
  simHitsEcalEnergyVsTimeHistEB_ = fileService->make<TH2F>("ecalEnergyVsTimeOfSimHitsEB","Energy vs. time of HSCP SimTrack-matching EB sim hits in event",115,-15,100,125,-1,4);
  simHitsEcalEnergyVsTimeHistEE_ = fileService->make<TH2F>("ecalEnergyVsTimeOfSimHitsEE","Energy vs. time of HSCP SimTrack-matching EE sim hits in event",115,-15,100,125,-1,4);
  simHitsEcalDigiMatchEnergyHistEB_ = fileService->make<TH1F>("ecalEnergyOfDigiMatSimHitsEB","HSCP digi-matching SimHit energy EB [GeV]",125,-1,4);
  simHitsEcalDigiMatchEnergyHistEE_ = fileService->make<TH1F>("ecalEnergyOfDigiMatSimHitsEE","HSCP digi-matching SimHit energy EE [GeV]",125,-1,4);
  simHitsEcalDigiMatchTimeHistEB_ = fileService->make<TH1F>("ecalTimingOfDigiMatSimHitsEB","HSCP digi-matching SimHit time EB [ns]",115,-15,100);
  simHitsEcalDigiMatchTimeHistEE_ = fileService->make<TH1F>("ecalTimingOfDigiMatSimHitsEE","HSCP digi-matching SimHit time EE [ns]",115,-15,100);
  simHitsEcalDigiMatchEnergyVsTimeHistEB_ = fileService->make<TH2F>("ecalEnergyVsTimeOfDigiMatSimHitsEB","HSCP digi-matching EB SimHit energy vs. time",115,-15,100,125,-1,4);
  simHitsEcalDigiMatchEnergyVsTimeHistEE_ = fileService->make<TH2F>("ecalEnergyVsTimeOfDigiMatSimHitsEE","HSCP digi-matching EE SimHit energy vs. time",115,-15,100,125,-1,4);
  simHitsEcalDigiMatchIEtaHist_ = fileService->make<TH1F>("ecalIEtaOfDigiMatchSimHits","iEta of digi-matching Ecal simHits (EB)",171,-85,86);
  simHitsEcalDigiMatchIPhiHist_ = fileService->make<TH1F>("ecalIPhiOfDigiMatchSimHits","iPhi of digi-matching Ecal simHits (EB)",360,1,361);
  digisEcalNumHistEB_ = fileService->make<TH1F>("ecalDigisNumberEB","Number of EB digis matching simhits in event",200,0,1000);
  digisEcalNumHistEE_ = fileService->make<TH1F>("ecalDigisNumberEE","Number of EE digis matching simhits in event",200,0,1000);
  digiOccupancyMapEB_ = fileService->make<TH2F>("ecalDigiOccupancyMapEB","Occupancy of simhit-matching digis EB;i#phi;i#eta",360,1,361,171,-85,86);
  digiOccupancyMapEEP_ = fileService->make<TH2F>("ecalDigiOccupancyMapEEM","Occupancy of simhit-matching digis EEM;ix;iy",100,1,100,100,1,100);
  digiOccupancyMapEEM_ = fileService->make<TH2F>("ecalDigiOccupancyMapEEP","Occupancy of simhit-matching digis EEP;ix;iy",100,1,100,100,1,100);

  // SIM-DIGI: RPC
  residualsRPCRecHitSimDigis_ = fileService->make<TH1F>("residualsRPCRecHitSimDigis","HSCP SimHit - Clossest RPC RecHit",100,-5,5);
  efficiencyRPCRecHitSimDigis_ = fileService->make<TH1F>("efficiencyRPCRecHitSimDigis","HSCP SimHits RecHits Efficiency",2,-0.5,1.5);
  cluSizeDistribution_ = fileService->make<TH1F>("RPCCluSizeDistro","RPC HSCP CluSize Distribution",11,-0.5,10.5);
  rpcTimeOfFlightBarrel_[0] = fileService->make<TH1F>("RPCToFLayer1","RPC HSCP Time Of Flight Layer 1",50,5,100);
  rpcTimeOfFlightBarrel_[1] = fileService->make<TH1F>("RPCToFLayer2","RPC HSCP Time Of Flight Layer 2",50,5,100);
  rpcTimeOfFlightBarrel_[2] = fileService->make<TH1F>("RPCToFLayer3","RPC HSCP Time Of Flight Layer 3",50,5,100);
  rpcTimeOfFlightBarrel_[3] = fileService->make<TH1F>("RPCToFLayer4","RPC HSCP Time Of Flight Layer 4",50,5,100);
  rpcTimeOfFlightBarrel_[4] = fileService->make<TH1F>("RPCToFLayer5","RPC HSCP Time Of Flight Layer 5",50,5,100);
  rpcTimeOfFlightBarrel_[5] = fileService->make<TH1F>("RPCToFLayer6","RPC HSCP Time Of Flight Layer 6",50,5,100);
  rpcBXBarrel_[0] = fileService->make<TH1F>("RPCBXLayer1","RPC HSCP BX Layer 1",5,-0.5,4.5);
  rpcBXBarrel_[1] = fileService->make<TH1F>("RPCBXLayer2","RPC HSCP BX Layer 2",5,-0.5,4.5);
  rpcBXBarrel_[2] = fileService->make<TH1F>("RPCBXLayer3","RPC HSCP BX Layer 3",5,-0.5,4.5);
  rpcBXBarrel_[3] = fileService->make<TH1F>("RPCBXLayer4","RPC HSCP BX Layer 4",5,-0.5,4.5);
  rpcBXBarrel_[4] = fileService->make<TH1F>("RPCBXLayer5","RPC HSCP BX Layer 5",5,-0.5,4.5);
  rpcBXBarrel_[5] = fileService->make<TH1F>("RPCBXLayer6","RPC HSCP BX Layer 6",5,-0.5,4.5);
  rpcTimeOfFlightEndCap_[0]= fileService->make<TH1F>("RPCToFDisk1","RPC HSCP Time Of Flight Disk 1",50,5,100);
  rpcTimeOfFlightEndCap_[1]= fileService->make<TH1F>("RPCToFDisk2","RPC HSCP Time Of Flight Disk 2",50,5,100);
  rpcTimeOfFlightEndCap_[2]= fileService->make<TH1F>("RPCToFDisk3","RPC HSCP Time Of Flight Disk 3",50,5,100);
  rpcBXEndCap_[0] = fileService->make<TH1F>("RPCBXDisk1","RPC HSCP BX Disk 1",5,-0.5,4.5);
  rpcBXEndCap_[1] = fileService->make<TH1F>("RPCBXDisk2","RPC HSCP BX Disk 2",5,-0.5,4.5);
  rpcBXEndCap_[2] = fileService->make<TH1F>("RPCBXDisk3","RPC HSCP BX Disk 3",5,-0.5,4.5);
}


HSCPValidator::~HSCPValidator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
//   particleEtaHist_ = fileService->make<TH1F>("particleEta","Eta of gen particle",100,-5,5);
//   particlePhiHist_ = fileService->make<TH1F>("particlePhi","Phi of gen particle",180,-3.15,3.15);
//   particlePHist_ = fileService->make<TH1F>("particleP","Momentum of gen particle",500,0,2000);
//   particlePtHist_ = fileService->make<TH1F>("particlePt","P_{T} of gen particle",500,0,2000);
//   particleMassHist_ = fileService->make<TH1F>("particleMass","Mass of gen particle",1000,0,2000);
//   particleStatusHist_ = fileService->make<TH1F>("particleStatus","Status of gen particle",10,0,10);
//   particleBetaHist_ = fileService->make<TH1F>("particleBeta","Beta of gen particle",100,0,1);
//   particleBetaInverseHist_ = fileService->make<TH1F>("particleBetaInverse","1/#beta of gen particle",100,0,5);

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HSCPValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);


  if(doGenPlots_)
    makeGenPlots(iEvent);
  if(doSimTrackPlots_)
    makeSimTrackPlots(iEvent);
  if(doSimDigiPlots_){
    makeSimDigiPlotsECAL(iEvent);
    makeSimDigiPlotsRPC(iEvent);
  }
  if(doHLTPlots_){
    makeHLTPlots(iEvent);
  }
  if(doRecoPlots_){
     makeRecoPlots(iEvent);
  }
}


// ------------ method called once each job just before starting event loop  ------------
void 
HSCPValidator::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCPValidator::endJob()
{
  std::string frequencies = "";
  for(std::map<int,int>::const_iterator itr = particleIdsFoundMap_.begin();
      itr != particleIdsFoundMap_.end(); ++itr)
  {
      frequencies+="PDG ID: ";
      frequencies+=intToString(itr->first);
      frequencies+=" Frequency: ";
      frequencies+=intToString(itr->second);
      frequencies+="\n";
  }
  std::cout << "Found PDGIds: " << "\n\n" << frequencies << std::endl;

}

// ------------- Make gen plots ---------------------------------------------------------
void HSCPValidator::makeGenPlots(const edm::Event& iEvent)
{
  using namespace edm;

  double missingpx=0;
  double missingpy=0;
  double missingpx_nohscp=0;
  double missingpy_nohscp=0;
  double scalorEt=0;
  double scalorEt_nohscp=0;


  Handle<HepMCProduct> evt;
  iEvent.getByLabel(label_, evt);

  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
  for(HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
      p != myGenEvent->particles_end(); ++p )
  {

    if((*p)->status() != particleStatus_)
      continue;
    //calculate MET(neutrino as MET)
    if(abs((*p)->pdg_id())!=12 && abs((*p)->pdg_id())!=14 && abs((*p)->pdg_id())!=16){ //for non-neutrino particles. 
       missingpx-=(*p)->momentum().px();
       missingpy-=(*p)->momentum().py();
       scalorEt+=(*p)->momentum().perp();
    }

    // Check if the particleId is in our R-hadron list
    std::vector<int>::const_iterator partIdItr = find(particleIds_.begin(),particleIds_.end(),(*p)->pdg_id());
    if(partIdItr==particleIds_.end()){
       
       //calculate MET(neutrino+ HSCP as MET)
       if(abs((*p)->pdg_id())!=12 && abs((*p)->pdg_id())!=14 && abs((*p)->pdg_id())!=16){ //for non-neutrino particles. 
          missingpx_nohscp-=(*p)->momentum().px();
          missingpy_nohscp-=(*p)->momentum().py();
          scalorEt_nohscp+=(*p)->momentum().perp();
        }
    }
    else{

       particleStatusHist_->Fill((*p)->status());
    
       std::pair<std::map<int,int>::iterator,bool> pair = particleIdsFoundMap_.insert(std::make_pair<int,int>((*p)->pdg_id(),1));
       if(!pair.second)
       {
          ++(pair.first->second);
       }
       
       double mag = sqrt(pow((*p)->momentum().px(),2) + pow((*p)->momentum().py(),2) + pow((*p)->momentum().pz(),2) );
       particleEtaHist_->Fill((*p)->momentum().eta());
       particlePhiHist_->Fill((*p)->momentum().phi());
       particlePHist_->Fill(mag);
       particlePtHist_->Fill((*p)->momentum().perp());
       particleMassHist_->Fill((*p)->generated_mass());
       float particleP = mag;
       float particleM = (*p)->generated_mass();
       particleBetaHist_->Fill(particleP/sqrt(particleP*particleP+particleM*particleM));
       particleBetaInverseHist_->Fill(sqrt(particleP*particleP+particleM*particleM)/particleP);
    }
       
  }

  h_genhscp_met->Fill(sqrt(missingpx*missingpx+missingpy*missingpy));
  h_genhscp_met_nohscp->Fill(sqrt(missingpx_nohscp*missingpx_nohscp+missingpy_nohscp*missingpy_nohscp));
  h_genhscp_scaloret->Fill(scalorEt);
  h_genhscp_scaloret_nohscp->Fill(scalorEt_nohscp);


  delete myGenEvent; 



}

// ------------- Make SimTrack plots ---------------------------------------------------------
void HSCPValidator::makeSimTrackPlots(const edm::Event& iEvent)
{  using namespace edm;
  //get sim track infos
  Handle<edm::SimTrackContainer> simTracksHandle;
  iEvent.getByLabel("g4SimHits",simTracksHandle);
  const SimTrackContainer simTracks = *(simTracksHandle.product());

  SimTrackContainer::const_iterator simTrack;

  for (simTrack = simTracks.begin(); simTrack != simTracks.end(); ++simTrack){
     // Check if the particleId is in our list
     std::vector<int>::const_iterator partIdItr = find(particleIds_.begin(),particleIds_.end(),simTrack->type());
     if(partIdItr==particleIds_.end()) continue;

     simTrackParticleEtaHist_->Fill((*simTrack).momentum().eta());
     simTrackParticlePhiHist_->Fill((*simTrack).momentum().phi());
     simTrackParticlePHist_->Fill((*simTrack).momentum().P());
     
     simTrackParticlePtHist_->Fill((*simTrack).momentum().pt());
     
     simTrackParticleBetaHist_->Fill((*simTrack).momentum().P()/(*simTrack).momentum().e());  
     
     // std::cout<<"Particle:"<<simTrack->type()<<" Charge:"<<simTrack->charge()<<std::endl;

  }
}
// ------------- Make HLT plots ---------------------------------------------------------
void HSCPValidator::makeHLTPlots(const edm::Event& iEvent)
{
  using namespace edm;
  //get HLT infos
     

      edm::TriggerResultsByName tr = iEvent.triggerResultsByName("HLT");

          if(!tr.isValid()){
        std::cout<<"Tirgger Results not available"<<std::endl;
      }
 
   edm::Handle< trigger::TriggerEvent > trEvHandle;
   iEvent.getByLabel("hltTriggerSummaryAOD", trEvHandle);
   trigger::TriggerEvent trEv = *trEvHandle;


   unsigned int TrIndex_Unknown     = tr.size();


   // HLT TRIGGER BASED ON 1 MUON!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_v1")){
      if(tr.accept(tr.triggerIndex("HLT_Mu40_v1"))) hltmu->Fill(1);
      else {hltmu->Fill(0);}
   }
   else{
      if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu30_v1")){
         if(IncreasedTreshold(trEv, InputTag("hltSingleMu30L3Filtered30","","HLT"), 40,2.1, 1, false)) hltmu->Fill(1);
         else hltmu->Fill(0);
      }else{
         printf("BUG with HLT_Mu\n");
         std::cout<<"trigger names are : ";
         for(unsigned int i=0;i<tr.size();i++){
            std::cout<<" "<<tr.triggerName(i);
         }
         std::cout<<std::endl;       
      }
   }


 // HLT TRIGGER BASED ON MET!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v3")){
      if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v3")))hltmet->Fill(1);
      else hltmet->Fill(0);
   }else{
      if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v2")){
          if(tr.accept(tr.triggerIndex("HLT_PFMHT150_v2"))) hltmet->Fill(1);
          else hltmet->Fill(0);
      }
      else{
         printf("BUG with HLT_MET\n");
         
      }
   }
   

  // HLT TRIGGER BASED ON 1 JET!
   if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet370_v1")){
       if(tr.accept(tr.triggerIndex("HLT_Jet370_v1")))hltjet->Fill(1);
       else   hltjet->Fill(0);
   }else{ 
      if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet100U")){
         if(IncreasedTreshold(trEv, InputTag("hlt1jet100U","","HLT"), 140, 5.,1, false))hltjet->Fill(1);
         else   hltjet->Fill(0);
      }else{
         if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet70U")){   
            if(IncreasedTreshold(trEv, InputTag("hlt1jet70U","","HLT"), 140, 5.,1, false))hltjet->Fill(1);
            else   hltjet->Fill(0);
         }else{
            if(TrIndex_Unknown != tr.triggerIndex("HLT_Jet50U")){
               if(IncreasedTreshold(trEv, InputTag("hlt1jet50U","","HLT"), 140,2.5, 1, false))hltjet->Fill(1);
               else   hltjet->Fill(0); 
            }else{
               printf("BUG with HLT_Jet\n");
               
            }
         }
      }
   }
   



  
}

// ------------- Make simDigi plots ECAL ------------------------------------------------
void HSCPValidator::makeSimDigiPlotsECAL(const edm::Event& iEvent)
{
  using namespace edm;
  // EB SimHits
  Handle<PCaloHitContainer> ebSimHits;
  iEvent.getByLabel(ebSimHitTag_, ebSimHits);
  if(!ebSimHits.isValid())
  {
    std::cout << "Cannot get EBSimHits from event!" << std::endl;
    return;
  }
  // EE SimHits
  Handle<PCaloHitContainer> eeSimHits;
  iEvent.getByLabel(eeSimHitTag_, eeSimHits);
  if(!eeSimHits.isValid())
  {
    std::cout << "Cannot get EESimHits from event!" << std::endl;
    return;
  }
  // SimTracks
  Handle<SimTrackContainer> simTracks;
  iEvent.getByLabel(simTrackTag_,simTracks);
  if(!simTracks.isValid())
  {
    std::cout << "Cannot get SimTracks from event!" << std::endl;
    return;
  }
  // EB Digis
  Handle<EBDigiCollection> ebDigis;
  iEvent.getByLabel(EBDigiCollection_,ebDigis);
  if(!ebDigis.isValid())
  {
    std::cout << "Cannot get EBDigis from event!" << std::endl;
    return;
  }
  // EE Digis
  Handle<EEDigiCollection> eeDigis;
  iEvent.getByLabel(EEDigiCollection_,eeDigis);
  if(!eeDigis.isValid())
  {
    std::cout << "Cannot get EEDigis from event!" << std::endl;
    return;
  }

  // EB first
  // 1) Look at SimTracks, getting only the HSCP tracks
  // 2) Match to PCaloHits
  // 3) Match to digis
  int numMatchedSimHitsEventEB = 0;
  int numMatchedDigisEventEB = 0;
  const PCaloHitContainer* phitsEB=0;
  phitsEB = ebSimHits.product();
  for(SimTrackContainer::const_iterator simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack)
  {
    // Check if the particleId is in our list
    std::vector<int>::const_iterator partIdItr = find(particleIds_.begin(),particleIds_.end(),simTrack->type());
    if(partIdItr==particleIds_.end())
      continue;

    PCaloHitContainer mySimHitsEB;
    std::vector<EBDataFrame> myDigisEB;

    //int particleId = simTrack->type();
    int trackId = simTrack->trackId();
    PCaloHitContainer::const_iterator simHitItr = phitsEB->begin();
    while(simHitItr != phitsEB->end())
    {
      if(simHitItr->geantTrackId()==trackId)
        mySimHitsEB.push_back(*simHitItr);
      ++simHitItr;
    }
    if(mySimHitsEB.size()==0)
    {
      std::cout << "Could not find matching EB PCaloHits for SimTrack id: " << trackId << ".  Skipping this SimTrack" << std::endl;
      continue;
    }

    // Loop over matching PCaloHits
    for(simHitItr = mySimHitsEB.begin(); simHitItr != mySimHitsEB.end(); ++simHitItr)
    {
      simHitsEcalEnergyHistEB_->Fill(simHitItr->energy());
      simHitsEcalTimeHistEB_->Fill(simHitItr->time());
      simHitsEcalEnergyVsTimeHistEB_->Fill(simHitItr->time(),simHitItr->energy());
      EBDetId simHitId = EBDetId(simHitItr->id());
      std::cout << "SimHit DetId found: " << simHitId << " for PDGid: " << simTrack->type() << std::endl;
      //std::cout << "SimHit hashedIndex: " << simHitId.hashedIndex() << std::endl;
      std::cout << "SimHit energy: " << simHitItr->energy() << " time: " << simHitItr->time() << std::endl;
      ++numMatchedSimHitsEventEB;

      EBDigiCollection::const_iterator digiItr = ebDigis->begin();
      while(digiItr != ebDigis->end() && (digiItr->id() != simHitId))
        ++digiItr;
      if(digiItr==ebDigis->end())
      {
        // Commented out for debugging ease, Aug 3 2009
        std::cout << "Could not find simHit detId: " << simHitId << "in EBDigiCollection!" << std::endl;
        continue;
      }
      std::vector<EBDataFrame>::const_iterator myDigiItr = myDigisEB.begin();
      while(myDigiItr != myDigisEB.end() && (digiItr->id() != myDigiItr->id()))
        ++myDigiItr;
      if(myDigiItr!=myDigisEB.end())
        continue; // if this digi is already in the list, skip it

      ++numMatchedDigisEventEB;
      EBDataFrame df = *digiItr;
      myDigisEB.push_back(df);
      std::cout << "SAMPLE ADCs: " << "\t";
      for(int i=0; i<10;++i)
        std::cout << i << "\t";
      std::cout << std::endl << "\t\t";
      for(int i=0; i < df.size(); ++i)
      {
        std::cout << df.sample(i).adc() << "\t";
      }
      std::cout << std::endl << std::endl;

      simHitsEcalDigiMatchEnergyHistEB_->Fill(simHitItr->energy());
      simHitsEcalDigiMatchTimeHistEB_->Fill(simHitItr->time());
      simHitsEcalDigiMatchEnergyVsTimeHistEB_->Fill(simHitItr->time(),simHitItr->energy());
      simHitsEcalDigiMatchIEtaHist_->Fill(((EBDetId)digiItr->id()).ieta());
      simHitsEcalDigiMatchIPhiHist_->Fill(((EBDetId)digiItr->id()).iphi());
      digiOccupancyMapEB_->Fill(((EBDetId)digiItr->id()).iphi(),((EBDetId)digiItr->id()).ieta());
    }
  }
  simHitsEcalNumHistEB_->Fill(numMatchedSimHitsEventEB);
  digisEcalNumHistEB_->Fill(numMatchedDigisEventEB);

  // EE next
  int numMatchedSimHitsEventEE = 0;
  int numMatchedDigisEventEE = 0;
  const PCaloHitContainer* phitsEE=0;
  phitsEE = eeSimHits.product();
  for(SimTrackContainer::const_iterator simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack)
  {
    // Check if the particleId is in our list
    std::vector<int>::const_iterator partIdItr = find(particleIds_.begin(),particleIds_.end(),simTrack->type());
    if(partIdItr==particleIds_.end())
      continue;

    PCaloHitContainer mySimHitsEE;
    std::vector<EEDataFrame> myDigisEE;

    //int particleId = simTrack->type();
    int trackId = simTrack->trackId();
    PCaloHitContainer::const_iterator simHitItr = phitsEE->begin();
    while(simHitItr != phitsEE->end())
    {
      if(simHitItr->geantTrackId()==trackId)
        mySimHitsEE.push_back(*simHitItr);
      ++simHitItr;
    }
    if(mySimHitsEE.size()==0)
    {
      std::cout << "Could not find matching EE PCaloHits for SimTrack id: " << trackId << ".  Skipping this SimTrack" << std::endl;
      continue;
    }

    // Loop over matching PCaloHits
    for(simHitItr = mySimHitsEE.begin(); simHitItr != mySimHitsEE.end(); ++simHitItr)
    {
      simHitsEcalEnergyHistEE_->Fill(simHitItr->energy());
      simHitsEcalTimeHistEE_->Fill(simHitItr->time());
      simHitsEcalEnergyVsTimeHistEE_->Fill(simHitItr->time(),simHitItr->energy());
      EEDetId simHitId = EEDetId(simHitItr->id());
      std::cout << "SimHit DetId found: " << simHitId << " for PDGid: " << simTrack->type() << std::endl;
      //std::cout << "SimHit hashedIndex: " << simHitId.hashedIndex() << std::endl;
      std::cout << "SimHit energy: " << simHitItr->energy() << " time: " << simHitItr->time() << std::endl;
      ++numMatchedSimHitsEventEE;

      EEDigiCollection::const_iterator digiItr = eeDigis->begin();
      while(digiItr != eeDigis->end() && (digiItr->id() != simHitId))
        ++digiItr;
      if(digiItr==eeDigis->end())
      {
        // Commented out for debugging ease, Aug 3 2009
        std::cout << "Could not find simHit detId: " << simHitId << "in EEDigiCollection!" << std::endl;
        continue;
      }
      std::vector<EEDataFrame>::const_iterator myDigiItr = myDigisEE.begin();
      while(myDigiItr != myDigisEE.end() && (digiItr->id() != myDigiItr->id()))
        ++myDigiItr;
      if(myDigiItr!=myDigisEE.end())
        continue; // if this digi is already in the list, skip it

      ++numMatchedDigisEventEE;
      EEDataFrame df = *digiItr;
      myDigisEE.push_back(df);
      std::cout << "SAMPLE ADCs: " << "\t";
      for(int i=0; i<10;++i)
        std::cout << i << "\t";
      std::cout << std::endl << "\t\t";
      for(int i=0; i < df.size(); ++i)
      {
        std::cout << df.sample(i).adc() << "\t";
      }
      std::cout << std::endl << std::endl;

      simHitsEcalDigiMatchEnergyHistEE_->Fill(simHitItr->energy());
      simHitsEcalDigiMatchTimeHistEE_->Fill(simHitItr->time());
      simHitsEcalDigiMatchEnergyVsTimeHistEE_->Fill(simHitItr->time(),simHitItr->energy());
      if(((EEDetId)digiItr->id()).zside() > 0)
        digiOccupancyMapEEP_->Fill(((EEDetId)digiItr->id()).ix(),((EEDetId)digiItr->id()).iy());
      else if(((EEDetId)digiItr->id()).zside() < 0)
        digiOccupancyMapEEM_->Fill(((EEDetId)digiItr->id()).ix(),((EEDetId)digiItr->id()).iy());
    }
  }
  simHitsEcalNumHistEE_->Fill(numMatchedSimHitsEventEE);
  digisEcalNumHistEE_->Fill(numMatchedDigisEventEE);

}
// ------------- Make Reco plots ---------------------------------------------------------
void HSCPValidator::makeRecoPlots(const edm::Event& iEvent)
{
  using namespace edm;
   using namespace reco;

  Handle<HepMCProduct> evt;
  iEvent.getByLabel(label_, evt);

  Handle<TrackCollection> tkTracks;
  iEvent.getByLabel("generalTracks",tkTracks);
  const reco::TrackCollection tkTC = *(tkTracks.product());
  
  Handle<ValueMap<DeDxData> >          dEdxTrackHandle;
  iEvent.getByLabel("dedxHarmonic2", dEdxTrackHandle);
  const ValueMap<DeDxData> dEdxTrack = *dEdxTrackHandle.product();
  
  for(size_t i=0; i<tkTracks->size(); i++){
     
     reco::TrackRef trkRef = reco::TrackRef(tkTracks, i);
        
     if(trkRef->pt()<5 || trkRef->normalizedChi2()>10) continue;
 
     double minR= 999;
     double hscpgenPt =-1;
     
     HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
     for(HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
         p != myGenEvent->particles_end(); ++p )
     {
        
        if((*p)->status() != particleStatus_)
           continue;
        // Check if the particleId is in our R-hadron list
        std::vector<int>::const_iterator partIdItr = find(particleIds_.begin(),particleIds_.end(),(*p)->pdg_id());
        if(partIdItr!=particleIds_.end()){
           
           //calculate DeltaR
           double distance =pow((*p)->momentum().eta()-trkRef->eta(),2)+pow((*p)->momentum().phi()-trkRef->phi(),2);
           distance =sqrt(distance);
           if(distance <minR ){
              minR = distance;
              hscpgenPt= (*p)->momentum().perp();
           }
        }
     }
     RecoHSCPPtVsGenPt->Fill(trkRef->pt(),hscpgenPt);
 
     delete myGenEvent;     
     double dedx = dEdxTrack[trkRef].dEdx();
     dedxVsp->Fill( trkRef->p(),dedx);
       
  }  

}

// ------------- Make simDigi plots RPC -------------------------------------------------
void HSCPValidator::makeSimDigiPlotsRPC(const edm::Event& iEvent)
{
  using namespace edm;

  //std::cout << " Getting the SimHits " <<std::endl;
  std::vector<Handle<edm::PSimHitContainer> > theSimHitContainers;
  iEvent.getManyByType(theSimHitContainers);
  //std::cout << " The Number of sim Hits is  " << theSimHitContainers.size() <<std::endl;

  Handle<RPCRecHitCollection> rpcRecHits;
  iEvent.getByLabel("rpcRecHits","",rpcRecHits);

  //SimTrack Stuff
  std::vector<PSimHit> theSimHits;

  for (int i = 0; i < int(theSimHitContainers.size()); i++){
    theSimHits.insert(theSimHits.end(),theSimHitContainers.at(i)->begin(),theSimHitContainers.at(i)->end());
  } 


  for (std::vector<PSimHit>::const_iterator iHit = theSimHits.begin(); iHit != theSimHits.end(); iHit++){

    std::vector<int>::const_iterator partIdItr = find(particleIds_.begin(),particleIds_.end(),(*iHit).particleType());
    if(partIdItr==particleIds_.end())
      continue;

    DetId theDetUnitId((*iHit).detUnitId());

    DetId simdetid= DetId((*iHit).detUnitId());

    if(simdetid.det()==DetId::Muon &&  simdetid.subdetId()== MuonSubdetId::RPC){//Only RPCs

      RPCDetId rollId(theDetUnitId);
      RPCGeomServ rpcsrv(rollId);

      //std::cout << " Reading the Roll"<<std::endl;
      const RPCRoll* rollasociated = rpcGeo->roll(rollId);

      //std::cout << " Getting the Surface"<<std::endl;
      const BoundPlane & RPCSurface = rollasociated->surface(); 

      GlobalPoint SimHitInGlobal = RPCSurface.toGlobal((*iHit).localPosition());

      std::cout<<"\t\t We have an RPC Sim Hit! in t="<<(*iHit).timeOfFlight()<<"ns "<<rpcsrv.name()<<" Global postition="<<SimHitInGlobal<<std::endl;

      int layer = 0;

      if(rollId.station()==1&&rollId.layer()==1) layer = 1;
      else if(rollId.station()==1&&rollId.layer()==2) layer = 2;
      else if(rollId.station()==2&&rollId.layer()==1) layer = 3;
      else if(rollId.station()==2&&rollId.layer()==2)  layer = 4;
      else if(rollId.station()==3) layer = 5;
      else if(rollId.station()==4) layer = 6;

      if(rollId.region()==0){
        rpcTimeOfFlightBarrel_[layer-1]->Fill((*iHit).timeOfFlight());
      }else{
        rpcTimeOfFlightEndCap_[rollId.station()-1]->Fill((*iHit).timeOfFlight());
      }

      std::cout<<"\t\t r="<<SimHitInGlobal.mag()<<" phi="<<SimHitInGlobal.phi()<<" eta="<<SimHitInGlobal.eta()<<std::endl;

      int cluSize = 0;
      int bx = 100;
      float minres = 3000.;

      std::cout<<"\t \t \t \t Getting RecHits in Roll Asociated"<<std::endl;

      typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
      rangeRecHits recHitCollection =  rpcRecHits->get(rollId);
      RPCRecHitCollection::const_iterator recHit;

      efficiencyRPCRecHitSimDigis_->Fill(0);

      for(recHit = recHitCollection.first; recHit != recHitCollection.second ; recHit++){
        LocalPoint recHitPos=recHit->localPosition();
        float res=(*iHit).localPosition().x()- recHitPos.x();
        if(fabs(res)<fabs(minres)){
          minres=res;
          cluSize = recHit->clusterSize();
          bx=recHit->BunchX();
          std::cout<<"\t New Min Res "<<res<<"cm."<<std::endl;
        }
      }

      if(minres<3000.){
        residualsRPCRecHitSimDigis_->Fill(minres);
        efficiencyRPCRecHitSimDigis_->Fill(1);
        cluSizeDistribution_->Fill(cluSize);
        if(rollId.region()==0) rpcBXBarrel_[layer-1]->Fill(bx);
        else rpcBXEndCap_[rollId.station()-1]->Fill(bx);
      }
    }
  }
}

// ------------- Convert int to string for printing -------------------------------------
std::string HSCPValidator::intToString(int num)
{
  using namespace std;
  ostringstream myStream;
  myStream << num << flush;
  return(myStream.str()); //returns the string form of the stringstream object
}



//------Increase trigger thresold----


bool HSCPValidator::IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, double etaCut, int NObjectAboveThreshold, bool averageThreshold)
{
   unsigned int filterIndex = trEv.filterIndex(InputPath);
   //if(filterIndex<trEv.sizeFilters())printf("SELECTED INDEX =%i --> %s    XXX   %s\n",filterIndex,trEv.filterTag(filterIndex).label().c_str(), trEv.filterTag(filterIndex).process().c_str());

   if (filterIndex<trEv.sizeFilters()){
      const trigger::Vids& VIDS(trEv.filterIds(filterIndex));
      const trigger::Keys& KEYS(trEv.filterKeys(filterIndex));
      const int nI(VIDS.size());
      const int nK(KEYS.size());
      assert(nI==nK);
      const int n(std::max(nI,nK));
      const trigger::TriggerObjectCollection& TOC(trEv.getObjects());


      if(!averageThreshold){
         int NObjectAboveThresholdObserved = 0;
         for (int i=0; i!=n; ++i) {
	   if(TOC[KEYS[i]].pt()> NewThreshold && fabs(TOC[KEYS[i]].eta())<etaCut) NObjectAboveThresholdObserved++;
            //cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "<< TOC[KEYS[i]].id() << " " << TOC[KEYS[i]].pt() << " " << TOC[KEYS[i]].eta() << " " << TOC[KEYS[i]].phi() << " " << TOC[KEYS[i]].mass()<< endl;
         }
         if(NObjectAboveThresholdObserved>=NObjectAboveThreshold)return true;

      }else{
         std::vector<double> ObjPt;

         for (int i=0; i!=n; ++i) {
            ObjPt.push_back(TOC[KEYS[i]].pt());
            //cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "<< TOC[KEYS[i]].id() << " " << TOC[KEYS[i]].pt() << " " << TOC[KEYS[i]].eta() << " " << TOC[KEYS[i]].phi() << " " << TOC[KEYS[i]].mass()<< endl;
         }
         if((int)(ObjPt.size())<NObjectAboveThreshold)return false;
         std::sort(ObjPt.begin(), ObjPt.end());

         double Average = 0;
         for(int i=0; i<NObjectAboveThreshold;i++){
            Average+= ObjPt[ObjPt.size()-1-i];
         }Average/=NObjectAboveThreshold;
         //cout << "AVERAGE = " << Average << endl;

         if(Average>NewThreshold)return true;
      }
   }
   return false;
}


//define this as a plug-in
DEFINE_FWK_MODULE(HSCPValidator);
