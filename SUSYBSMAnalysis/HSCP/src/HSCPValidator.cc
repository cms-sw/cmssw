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
// $Id: HSCPValidator.cc,v 1.3 2010/04/15 12:57:39 carrillo Exp $
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
  doSimDigiPlots_ (iConfig.getParameter<bool>("MakeSimDigiPlots")),
  doRecoPlots_ (iConfig.getParameter<bool>("MakeRecoPlots")),
  label_ (iConfig.getParameter<edm::InputTag>("generatorLabel")),
  particleIds_ (iConfig.getParameter< std::vector<int> >("particleIds")),
  particleStatus_ (iConfig.getUntrackedParameter<int>("particleStatus",3)),
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
  if(doSimDigiPlots_){
    makeSimDigiPlotsECAL(iEvent);
    makeSimDigiPlotsRPC(iEvent);
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

  Handle<HepMCProduct> evt;
  iEvent.getByLabel(label_, evt);

  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
  for(HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
      p != myGenEvent->particles_end(); ++p )
  {
    // Check if the particleId is in our list
    std::vector<int>::const_iterator partIdItr = find(particleIds_.begin(),particleIds_.end(),(*p)->pdg_id());
    if(partIdItr==particleIds_.end())
      continue;

    particleStatusHist_->Fill((*p)->status());

    if((*p)->status() != particleStatus_)
      continue;

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

    //std::cout << "FOUND PARTICLE WITH PDGid: " << (*p)->pdg_id() << std::endl;
    //std::cout << "FOUND PARTICLE in param array where its id is " << particleID[i] << std::endl;
    //std::cout << "\tParticle -- eta: " << (*p)->momentum().eta() << std::endl;

    //if((*p)->momentum().perp() > ptMin[i] && (*p)->momentum().eta() > etaMin[i] 
    //    && (*p)->momentum().eta() < etaMax[i] && ((*p)->status() == status[i] || status[i] == 0))
    //{
    //  std::cout << "!!!!PARTICLE ACCEPTED" << std::endl;
    //}  
  }
  delete myGenEvent; 
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

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPValidator);
