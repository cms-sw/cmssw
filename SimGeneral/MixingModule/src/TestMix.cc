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
// $Id: TestMix.cc,v 1.10 2006/07/18 16:53:03 uberthon Exp $
//
//


// system include files
#include <memory>
#include <utility>

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
    std::cout<<"\n=================== Starting SimHit access, subdet "<<subdet<<"  ==================="<<std::endl;
    std::auto_ptr<MixCollection<PSimHit> > col(new MixCollection<PSimHit>(cf.product(), subdet,std::pair<int,int>(-1,2)));
    std::cout<<*(col.get())<<std::endl;
    MixCollection<PSimHit>::iterator cfi;
    int count=0;
    for (cfi=col->begin(); cfi!=col->end();cfi++) {
      std::cout<<" Hit "<<count<<" has tof "<<cfi->timeOfFlight()<<" trackid "<<cfi->trackId() <<" bunchcr "<<cfi.bunch()<<" trigger "<<cfi.getTrigger()<<", from EncodedEventId: "<<cfi->eventId().bunchCrossing()<<" " <<cfi->eventId().event() <<std::endl;
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
    std::cout <<" \nFor all containers we got "<<all_trackhits->sizeSignal()<<" signal hits and "<<all_trackhits->sizePileup()<<" pileup hits, total: "<<all_trackhits->size()<<std::endl;
    MixCollection<PSimHit>::iterator it;
    int ii=0;
    for (it=all_trackhits->begin(); it!= all_trackhits->end();it++) {
      std::cout<<" Hit "<<ii<<" of all hits has tof "<<it->timeOfFlight()<<" trackid "<<it->trackId() <<" bunchcr "<<it.bunch()<<" trigger "<<it.getTrigger()<<", from EncodedEventId: "<<it->eventId().bunchCrossing() <<" "<<it->eventId().event()<<std::endl;
      ii++;
    }
}

