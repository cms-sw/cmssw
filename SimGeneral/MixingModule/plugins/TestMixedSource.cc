// -*- C++ -*-
//
// Package:    TestMixedSource
// Class:      TestMixedSource
// 
/**\class TestMixedSource TestMixedSource.cc TestMixed/TestMixedSource/src/TestMixedSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Emilia Lubenova Becheva
//         Created:  Wed May 20 16:46:58 CEST 2009
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TH1I.h"
#include "TFile.h"

#include "TestMixedSource.h"

#include <iostream>
#include <fstream>


//
// constructors and destructor
//
namespace edm
{
TestMixedSource::TestMixedSource(const edm::ParameterSet& iConfig)
: fileName_(iConfig.getParameter<std::string>("fileName")), minbunch_(iConfig.getParameter<int>("minBunch")),maxbunch_(iConfig.getParameter<int>("maxBunch"))
{ 
  
  histTrack_bunchSignal_ = new TH1I("histoTrackSignal","Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
  histTrack_bunchPileups_ = new TH1I("histoTrackPileups","Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);

  // Vertex
  histVertex_bunch_ = new TH1I("histoVertex","Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
  
  // PCaloHit
  histPCaloHit_bunch_ = new TH1I("histoPCaloHit","Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
  
  // PSimHit
  histPSimHit_bunchSignal_TrackerHitsTECHighTof_ = new TH1I("histoPSimHitTrackerHitsTECHighTofSignal","Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
  histPSimHit_bunchPileups_TrackerHitsTECHighTof_ = new TH1I("histoPSimHitTrackerHitsTECHighTofPileups","Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
  histPSimHit_bunchSignal_MuonCSCHits_ = new TH1I("histoPSimHitMuonCSCHitsSignal","Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
  histPSimHit_bunchPileups_MuonCSCHits_ = new TH1I("histoPSimHitMuonCSCHitsPileups","Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);

  int bsp = 25;//bunchspace
  tofhist_ = new TH1I ("TrackerHit_Tof_bcr","TrackerHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
  tofhist_sig_ = new TH1I("SignalTrackerHit_Tof_bcr","TrackerHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);


  // HepMCProduct
  histHepMCProduct_bunch_ = new TH1I("histoHepMCProduct","Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);

  // Tokens

  edm::InputTag tag = edm::InputTag("mix","g4SimHits");
  
  SimTrackToken_ = consumes<CrossingFrame<SimTrack>>(tag);
  SimVertexToken_ = consumes<CrossingFrame<SimVertex>>(tag);

  tag = edm::InputTag("mix","g4SimHitsTrackerHitsTECHighTof");
  TrackerToken0_ = consumes<CrossingFrame<PSimHit>>(tag);

  tag = edm::InputTag("mix","g4SimHitsEcalHitsEB");
  CaloToken1_ = consumes<CrossingFrame<PCaloHit>>(tag);

  tag = edm::InputTag("mix","g4SimHitsMuonCSCHits");
  MuonToken_ = consumes<CrossingFrame<PSimHit>>(tag);

  tag = edm::InputTag("mix","generatorSmeared");
  HepMCToken_ = consumes<CrossingFrame<edm::HepMCProduct>>(tag);


}


TestMixedSource::~TestMixedSource()
{ std::cout << " Destructor TestMixed"  << std::endl;

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TestMixedSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
  // test SimTracks
  //----------------------
  edm::Handle<CrossingFrame<SimTrack> > cf_simtrack;
  bool gotTracks = iEvent.getByToken(SimTrackToken_,cf_simtrack);
  if (!gotTracks)  outputFile<<" Could not read SimTracks!!!!" 
  		   << " Please, check if the object SimTracks has been declared in the"
  		   << " MixingModule configuration file."<<std::endl;
  
  if (gotTracks) {
    outputFile<<"\n=================== Starting SimTrack access ==================="<<std::endl;

    std::unique_ptr<MixCollection<SimTrack> > col1(new MixCollection<SimTrack>(cf_simtrack.product()));
    MixCollection<SimTrack>::iterator cfi1;
    int count1=0;
    std::cout <<" \nWe got "<<col1->sizeSignal()<<" signal tracks and "<<col1->sizePileup()<<" pileup tracks, total: "<<col1->size()<<std::endl;
    for (cfi1=col1->begin(); cfi1!=col1->end();cfi1++) {
      //std::cout << " BUNCH cfi1.bunch() = " << cfi1.bunch() << std::endl;
      if (cfi1.getTrigger()==0){
      histTrack_bunchPileups_->Fill(cfi1.bunch());
      }
      
      if (cfi1.getTrigger()==1){
      histTrack_bunchSignal_->Fill(cfi1.bunch());
      }

      
      int a = count1%4;
      if (a==3){
      outputFile<<" SimTrack "<<count1<<" has genpart index  "<<cfi1->genpartIndex()<<" vertex Index "<<cfi1->vertIndex() <<" bunchcr "<<cfi1.bunch()<<" trigger "<<cfi1.getTrigger()<<", from EncodedEventId: "<<cfi1->eventId().bunchCrossing() <<" "<<cfi1->eventId().event() <<std::endl;
      }
      count1++; 
    }
  }

  
  // test SimVertices
  //---------------------
  edm::Handle<CrossingFrame<SimVertex> > cf_simvtx;
  bool gotSimVertex = iEvent.getByToken(SimVertexToken_,cf_simvtx);
  if (!gotSimVertex) outputFile<<" Could not read Simvertices !!!!"<<std::endl;
  else {
    outputFile<<"\n=================== Starting SimVertex access ==================="<<std::endl;
    std::unique_ptr<MixCollection<SimVertex> > col2(new MixCollection<SimVertex>(cf_simvtx.product()));
    MixCollection<SimVertex>::iterator cfi2;
    int count2=0;
    outputFile <<" \nWe got "<<col2->sizeSignal()<<" signal vertices and "<<col2->sizePileup()<<" pileup vertices, total: "<<col2->size()<<std::endl;
    for (cfi2=col2->begin(); cfi2!=col2->end();cfi2++) {
      histVertex_bunch_->Fill(cfi2.bunch());
      int b = count2%4;
      if (count2 == 0 || b==3){
      //outputFile<<" SimVertex "<<count2<<" has parent index  "<<cfi2->parentIndex()<<" bunchcr "<<cfi2.bunch()<<" trigger "<<cfi2.getTrigger()<<", from EncodedEventId: "<<cfi2->eventId().bunchCrossing() <<" "<<cfi2->eventId().event() <<std::endl;
      }
      //SimVertex myvtx=(*cfi2);
      //outputFile<<"Same with op*: "<<count2<<" has parent index  "<<myvtx.parentIndex()<<" bunchcr "<<cfi2.bunch()<<" trigger "<<cfi2.getTrigger()<<", from EncodedEventId: "<<myvtx.eventId().bunchCrossing() <<" "<<myvtx.eventId().event() <<std::endl;
      count2++; 
    }
  }
  
  
  //test HepMCProducts
  //------------------------------------
  edm::Handle<CrossingFrame<edm::HepMCProduct> > cf_hepmc;
  bool gotHepMCP = iEvent.getByToken(HepMCToken_,cf_hepmc);
  if (!gotHepMCP) std::cout<<" Could not read HepMCProducts!!!!"<<std::endl;
  else {
    outputFile<<"\n=================== Starting HepMCProduct access ==================="<<std::endl;
    std::unique_ptr<MixCollection<edm::HepMCProduct> > colhepmc(new MixCollection<edm::HepMCProduct>(cf_hepmc.product()));
    MixCollection<edm::HepMCProduct>::iterator cfihepmc;
    
    int count3=0;
    outputFile <<" \nWe got "<<colhepmc->sizeSignal()<<" signal hepmc products and "<<colhepmc->sizePileup()<<" pileup hepmcs, total: "<<colhepmc->size()<<std::endl;
    for (cfihepmc=colhepmc->begin(); cfihepmc!=colhepmc->end();cfihepmc++) {
      histHepMCProduct_bunch_->Fill(cfihepmc.bunch());
      int c = count3%4;
      if (count3==0 || c==3){
      //outputFile<<" edm::HepMCProduct "<<count3<<" has event number "<<cfihepmc->GetEvent()->event_number()<<", "<< cfihepmc->GetEvent()->particles_size()<<" particles and "<<cfihepmc->GetEvent()->vertices_size()<<" vertices,  bunchcr= "<<cfihepmc.bunch()<<" trigger= "<<cfihepmc.getTrigger() <<" sourcetype= "<<cfihepmc.getSourceType()<<std::endl;
      }
      HepMCProduct myprod=colhepmc->getObject(count3);
      //outputFile<<"same with getObject:hepmc product   "<<count3<<" has event number "<<myprod.GetEvent()->event_number()<<", "<<myprod.GetEvent()->particles_size()<<" particles and "<<myprod.GetEvent()->vertices_size()<<" vertices"<<std::endl;
      count3++;
    }
  }


  // test CaloHits
  //--------------------------  
  const std::string subdetcalo("g4SimHitsEcalHitsEB");
  edm::Handle<CrossingFrame<PCaloHit> > cf_calo;
  bool gotPCaloHit = iEvent.getByToken(CaloToken1_,cf_calo);
  if (!gotPCaloHit) outputFile<<" Could not read CaloHits with label "<<subdetcalo<<"!!!!"<<std::endl;
  else {
    outputFile<<"\n\n=================== Starting CaloHit access, subdet "<<subdetcalo<<"  ==================="<<std::endl;
    std::unique_ptr<MixCollection<PCaloHit> > colcalo(new MixCollection<PCaloHit>(cf_calo.product()));
    //outputFile<<*(colcalo.get())<<std::endl;
    MixCollection<PCaloHit>::iterator cficalo;
    int count4=0;
    for (cficalo=colcalo->begin(); cficalo!=colcalo->end();cficalo++) {
      histPCaloHit_bunch_->Fill(cficalo.bunch());
      int d = count4%4;
      if (count4==0 || d==3){
      //outputFile<<" CaloHit "<<count4<<" has tof "<<cficalo->time()<<" trackid "<<cficalo->geantTrackId() <<" bunchcr "<<cficalo.bunch()<<" trigger "<<cficalo.getTrigger()<<", from EncodedEventId: "<<cficalo->eventId().bunchCrossing()<<" " <<cficalo->eventId().event() <<std::endl;
      }
      count4++;
    }
  }

  
  // test PSimHit for one particular subdet
  //---------------------------------------
  
  
  const std::string subdet("g4SimHitsTrackerHitsTECHighTof");
  edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
  bool gotPSimHit = iEvent.getByToken(TrackerToken0_,cf_simhit);
  if (!gotPSimHit) outputFile<<" Could not read SimHits with label "<<subdet<<"!!!!"<<std::endl;
  else {
    outputFile<<"\n\n=================== Starting SimHit access, subdet "<<subdet<<"  ==================="<<std::endl;

    std::unique_ptr<MixCollection<PSimHit> > col(new MixCollection<PSimHit>(cf_simhit.product()));
    //outputFile<<*(col.get())<<std::endl;
    MixCollection<PSimHit>::iterator cfi;
    int count5=0;
    for (cfi=col->begin(); cfi!=col->end();cfi++) {
      
      // Signal
      if (cfi.getTrigger()==1){ 
        histPSimHit_bunchSignal_TrackerHitsTECHighTof_->Fill(cfi.bunch());
        tofhist_sig_->Fill(cfi->timeOfFlight());
      }
      
      // Pileups
      if (cfi.getTrigger()==0){
       histPSimHit_bunchPileups_TrackerHitsTECHighTof_->Fill(cfi.bunch());
       std::cout << " cfi->timeOfFlight() = " << cfi->timeOfFlight() << std::endl;
       tofhist_->Fill(cfi->timeOfFlight()); 
      }
      
      int e = count5%4;
      if (e==3){
      outputFile<<" Hit "<<count5<<" has tof "<<cfi->timeOfFlight()<<" trackid "<<cfi->trackId() <<" bunchcr "<<cfi.bunch()<<" trigger "<<cfi.getTrigger()<<", from EncodedEventId: "<<cfi->eventId().bunchCrossing()<<" " <<cfi->eventId().event() <<" bcr from MixCol "<<cfi.bunch()<<std::endl;
      }
      count5++;
    }
  }
  
  const std::string subdet1("g4SimHitsMuonCSCHits");
  edm::Handle<CrossingFrame<PSimHit> > cf_simhit1;
  bool gotPSimHit1 = iEvent.getByToken(MuonToken_,cf_simhit1);
  if (!gotPSimHit1) outputFile<<" Could not read SimHits with label "<<subdet1<<"!!!!"<<std::endl;
  else {
    outputFile<<"\n\n=================== Starting SimHit access, subdet "<<subdet1<<"  ==================="<<std::endl;

    std::unique_ptr<MixCollection<PSimHit> > col(new MixCollection<PSimHit>(cf_simhit1.product()));
    //outputFile<<*(col.get())<<std::endl;
    MixCollection<PSimHit>::iterator cfi;
    int count5=0;
    for (cfi=col->begin(); cfi!=col->end();cfi++) {
      
      if (cfi.getTrigger()==1) histPSimHit_bunchSignal_MuonCSCHits_->Fill(cfi.bunch());
      
      if (cfi.getTrigger()==0) histPSimHit_bunchPileups_MuonCSCHits_->Fill(cfi.bunch());
      
      int e = count5%4;
      if (e==3){
      outputFile<<" Hit "<<count5<<" has tof "<<cfi->timeOfFlight()<<" trackid "<<cfi->trackId() <<" bunchcr "<<cfi.bunch()<<" trigger "<<cfi.getTrigger()<<", from EncodedEventId: "<<cfi->eventId().bunchCrossing()<<" " <<cfi->eventId().event() <<" bcr from MixCol "<<cfi.bunch()<<std::endl;
      }
      count5++;
    }
  }
  


}


// ------------ method called once each job just before starting event loop  ------------
void 
TestMixedSource::beginJob()
{ 
   outputFile.open("test.log");
   if (!outputFile.is_open())
   {
     std::cout << "Unable to open file!" << std::endl;
   } 
   
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestMixedSource::endJob() {
  
  outputFile.close();
  
  char t[30];
  sprintf(t,"%s",fileName_.c_str());
  std::cout << " fileName = " << t << std::endl; 
  histFile_=new TFile(t,"RECREATE");
  histTrack_bunchPileups_->Write();
  histTrack_bunchSignal_->Write();
  histVertex_bunch_->Write();
  histPCaloHit_bunch_->Write();
  histPSimHit_bunchPileups_TrackerHitsTECHighTof_->Write();
  histPSimHit_bunchSignal_TrackerHitsTECHighTof_->Write();
  tofhist_sig_->Write();
  tofhist_->Write();
  histPSimHit_bunchSignal_MuonCSCHits_->Write();
  histPSimHit_bunchPileups_MuonCSCHits_->Write();
  histHepMCProduct_bunch_->Write();
  histFile_->Write();
  histFile_->Close();

}
}//edm
