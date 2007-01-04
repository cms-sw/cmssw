// -*- C++ -*-
//
// Class:      TestMix
// 
/**\class TestMix

 Description: test of Mixing Module

*/
//
// Original Author:  Ursula Berthon
//         Created:  Fri Sep 23 11:38:38 CEST 2005
// $Id: TestMix.cc,v 1.15 2006/12/13 18:04:16 uberthon Exp $
//
//


// system include files
#include <memory>
#include <utility>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimGeneral/MixingModule/interface/TestMix.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

using namespace edm;

TestMix::TestMix(const edm::ParameterSet& iConfig): 
  level_(iConfig.getUntrackedParameter<int>("PrintLevel"))
{
  std::cout << "Constructed testMix , level "<<level_<<std::endl;

  track_containers_.push_back("TrackerHitsTECHighTof");
  track_containers_.push_back("TrackerHitsTECLowTof");

  track_containers2_.push_back("TrackerHitsTECLowTof");
  track_containers2_.push_back("TrackerHitsTECHighTof");

}


TestMix::~TestMix()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to analyze the data  ------------
void
TestMix::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

// Get input
    edm::Handle<CrossingFrame> cf;
    iEvent.getByType(cf);

    // and print
    std::cout<<std::endl;
    cf.product()->print(level_);

    // test accesses to CrossingFrame
    // attention: operator-> returns the templated object, but
    // bunch() and getTrigger() are methods of the iterator itself!

    // test access to SimHits
    const std::string subdet("TrackerHitsTECHighTof");
    std::cout<<"\n\n=================== Starting SimHit access, subdet "<<subdet<<"  ==================="<<std::endl;
    std::auto_ptr<MixCollection<PSimHit> > col(new MixCollection<PSimHit>(cf.product(), subdet,std::pair<int,int>(-1,2)));
    std::cout<<*(col.get())<<std::endl;
    MixCollection<PSimHit>::iterator cfi;
    int count=0;
    for (cfi=col->begin(); cfi!=col->end();cfi++) {
      std::cout<<" Hit "<<count<<" has tof "<<cfi->timeOfFlight()<<" trackid "<<cfi->trackId() <<" bunchcr "<<cfi.bunch()<<" trigger "<<cfi.getTrigger()<<", from EncodedEventId: "<<cfi->eventId().bunchCrossing()<<" " <<cfi->eventId().event() <<std::endl;
      count++;
     }

    // test access to CaloHits
    const std::string subdetcalo("EcalHitsEB");
    std::cout<<"\n\n=================== Starting CaloHit access, subdet "<<subdetcalo<<"  ==================="<<std::endl;
    std::auto_ptr<MixCollection<PCaloHit> > colcalo(new MixCollection<PCaloHit>(cf.product(), subdetcalo,std::pair<int,int>(-1,2)));
    std::cout<<*(colcalo.get())<<std::endl;
    MixCollection<PCaloHit>::iterator cficalo;
    count=0;
    for (cficalo=colcalo->begin(); cficalo!=colcalo->end();cficalo++) {
      std::cout<<" CaloHit "<<count<<" has tof "<<cficalo->time()<<" trackid "<<cficalo->geantTrackId() <<" bunchcr "<<cficalo.bunch()<<" trigger "<<cficalo.getTrigger()<<", from EncodedEventId: "<<cficalo->eventId().bunchCrossing()<<" " <<cficalo->eventId().event() <<std::endl;
      count++;
     }

    // test access to SimTracks
    std::cout<<"\n=================== Starting SimTrack access ==================="<<std::endl;
    std::auto_ptr<MixCollection<SimTrack> > col2(new MixCollection<SimTrack>(cf.product()));
    MixCollection<SimTrack>::iterator cfi2;
    int count2=0;
    std::cout <<" \nWe got "<<col2->sizeSignal()<<" signal tracks and "<<col2->sizePileup()<<" pileup tracks, total: "<<col2->size()<<std::endl;
    for (cfi2=col2->begin(); cfi2!=col2->end();cfi2++) {
      std::cout<<" SimTrack "<<count2<<" has genpart index  "<<cfi2->genpartIndex()<<" vertex Index "<<cfi2->vertIndex() <<" bunchcr "<<cfi2.bunch()<<" trigger "<<cfi2.getTrigger()<<", from EncodedEventId: "<<cfi2->eventId().bunchCrossing() <<" "<<cfi2->eventId().event() <<std::endl;
      count2++; 
    }

    // test access to SimVerticess
    std::cout<<"\n=================== Starting SimVertex access ==================="<<std::endl;
    std::auto_ptr<MixCollection<SimVertex> > col3(new MixCollection<SimVertex>(cf.product()));
    MixCollection<SimVertex>::iterator cfi3;
    int count3=0;
    std::cout <<" \nWe got "<<col3->sizeSignal()<<" signal vertices and "<<col3->sizePileup()<<" pileup vertices, total: "<<col3->size()<<std::endl;
    for (cfi3=col3->begin(); cfi3!=col3->end();cfi3++) {
      std::cout<<" SimVertex "<<count3<<" has parent index  "<<cfi3->parentIndex()<<" bunchcr "<<cfi3.bunch()<<" trigger "<<cfi3.getTrigger()<<", from EncodedEventId: "<<cfi3->eventId().bunchCrossing() <<" "<<cfi3->eventId().event() <<std::endl;
      count3++; 
    }

    //test MixCollection constructor with several subdetector names
    std::auto_ptr<MixCollection<PSimHit> > all_trackhits(new MixCollection<PSimHit>(cf.product(),track_containers_));
    std::cout<<"\n=================== Starting test for coll of several ROU-s ==================="<<std::endl;

    std::cout <<" \nFor all containers we got "<<all_trackhits->sizeSignal()<<" signal hits and "<<all_trackhits->sizePileup()<<" pileup hits, total: "<<all_trackhits->size()<<std::endl;
    MixCollection<PSimHit>::iterator it;
    int ii=0;
    for (it=all_trackhits->begin(); it!= all_trackhits->end();it++) {
      std::cout<<" Hit "<<ii<<" of all hits has tof "<<it->timeOfFlight()<<" trackid "<<it->trackId() <<" bunchcr "<<it.bunch()<<" trigger "<<it.getTrigger()<<", from EncodedEventId: "<<it->eventId().bunchCrossing() <<" "<<it->eventId().event()<<std::endl;
      ii++;
    }

    //test the same in different order: should be the same sizes, different order
    std::auto_ptr<MixCollection<PSimHit> > all_trackhits2(new MixCollection<PSimHit>(cf.product(),track_containers2_));
    std::cout <<" \nSame containers, different order: we got "<<all_trackhits2->sizeSignal()<<" signal hits and "<<all_trackhits2->sizePileup()<<" pileup hits, total: "<<all_trackhits2->size()<<std::endl;
    MixCollection<PSimHit>::iterator it2;
    int ii2=0;
    for (it2=all_trackhits2->begin(); it2!= all_trackhits2->end();it2++) {
      std::cout<<" Hit "<<ii2<<" of all hits has tof "<<it2->timeOfFlight()<<" trackid "<<it2->trackId() <<" bunchcr "<<it2.bunch()<<" trigger "<<it2.getTrigger()<<", from EncodedEventId: "<<it2->eventId().bunchCrossing() <<" "<<it2->eventId().event()<<std::endl;
      ii2++;
    }
    // for comparison
    std::cout<<" \nFor comparison: "<<std::endl;
    for (unsigned int j=0;j<track_containers_.size();++j) {
      std::auto_ptr<MixCollection<PSimHit> > trackhits(new MixCollection<PSimHit>(cf.product(),track_containers_[j]));
      std::cout <<track_containers_[j]<<" has "<< trackhits->sizeSignal()<<" signal Hits, and "<<trackhits->sizePileup() <<" pileup Hits, "<<", total "<< trackhits->size() <<std::endl;
    }
 
    // test reusage of the same iterator
   int ii3=0;
   for (it2=all_trackhits2->begin(); it2!= all_trackhits2->end();it2++) ii3++;
   if (ii3!=ii2) std::cout<<" Problem when re-using iterator!!"<<std::endl;
   else  std::cout<<" \nNo problem when re-using iterator."<<std::endl;

   // test access to non-filled collections
   //cases:   0) ok, collection has members
   //         1) bunchrange given outside of existent bunchcrossing numbers ==>exc
   //         2)collection not found in the registry of the input file ==> no exc
   //         3)for CaloHits/SimHits: template type does not correspond to subdetector given ==>exc
    std::cout<<"\n=================== Starting tests for abnormal conditions ==================="<<std::endl;

   // test case 0
      std::cout<<"\n[ Testing abnormal conditions case 0]Should be all ok: registry: "<<all_trackhits->inRegistry()<<" size: "<<all_trackhits->size()<<std::endl;

   // test case 1
   MixCollection<PSimHit> * col21=0;
   try {
     col21=new MixCollection<PSimHit>(cf.product(), subdet,std::pair<int,int>(-10,20));
   } catch ( cms::Exception &e ) { std::cout<<" [Testing abnormal conditions case2] exception bad runrange ok "<<std::endl; }


   // test case 2
   // first, on non-existing
   MixCollection<PSimHit> * col22=0;
   MixCollection<PSimHit>::iterator it22;
   std::string subdet_nonex("TrackerHitsTECHighToff");
   col22=new  MixCollection<PSimHit>(cf.product(), subdet_nonex,std::pair<int,int>(-1,2));
   std::cout<<" [Testing abnormal conditions case2]Non-existing subdetector: registry: "<<col22->inRegistry()<<" size: "<<col22->size()<<std::endl;
   int jj=0;
   for (it22=col22->begin(); it22!= col22->end();it22++) jj++;
   if (jj)       std::cout<<" [Testing abnormal conditions case2]Non-existing subdetector: iterator not ok!"<<std::endl;
   else  std::cout<<" [Testing abnormal conditions case2]Non-existing subdetector: iterator ok."<<std::endl;
   // second, vector
   //   const std::vector<std::string> subdet_nonexv("TrackerHitsTECHighToff","TrackerHitsTECLowTof");
   std::vector<std::string> subdet_nonexv(2);
   subdet_nonexv[0]="TrackerHitsTECHighToff";
   subdet_nonexv[1]="TrackerHitsTECLowTof";
   col22=new  MixCollection<PSimHit>(cf.product(), subdet_nonexv,std::pair<int,int>(-1,2));
   std::cout<<" [Testing abnormal conditions case2]Non-existing subdetector vector: registry: "<<col22->inRegistry()<<" size: "<<col22->size()<<std::endl;
   jj=0;
   for (it22=col22->begin(); it22!= col22->end();it22++) jj++;
   std::cout<<" [Testing abnormal conditions case2]Non-existing subdetector vector: iterator gives "<<jj<<" elements"<<std::endl;

   // test case 3
   MixCollection<PCaloHit>*  col23=0;
   try {
     col23=new MixCollection<PCaloHit>(cf.product(), subdet,std::pair<int,int>(-1,2));
   } catch ( cms::Exception &e ) { std::cout<<" [Testing abnormal conditions case3]bad template type exception ok"<<std::endl; }

<<<<<<< TestMix.cc
   // test getSignal
    std::cout<<"\n\n=================== Starting signal access  ==================="<<std::endl;
   const std::vector<SimTrack> *v;
   cf->getSignal("",v);
   std::cout<<"There are "<<v->size()<<" SimTracks "<<std::endl;
=======
   // test getters per bunchcrossing
   std::vector<SimTrack> v;
   v=cf->getPileupTracks(999);   
   v=cf->getPileupTracks(0);
   if (v.size()!=cf->getNrPileupTracks(0)) std::cout<<" [Testing result  of getPileups(..,bcr)] bad results: is: "<<v.size()<<" should be  "<<cf->getNrPileupTracks(0)<<std::endl;
   SimTrack t=v[0];
   std::cout<<" track vertex index: "<<t.vertIndex()<<std::endl;
>>>>>>> 1.15
}

