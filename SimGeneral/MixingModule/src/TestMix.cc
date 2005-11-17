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
// $Id: TestMix.cc,v 1.3 2005/10/27 08:43:40 uberthon Exp $
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
  level_(iConfig.getParameter<int>("PrintLevel"))
{
  std::cout << "Constructed testMix , level "<<level_<<std::endl;


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

    // test access to SimHits
    const std::string subdet("TrackerHitsTOBLowTof");
    std::cout<<"\n=================== Starting SimHit access, subdet "<<subdet<<"  ==================="<<std::endl;
    std::auto_ptr<MixCollection<PSimHit> > col(new MixCollection<PSimHit>(cf.product(), subdet));
    MixCollection<PSimHit>::iterator cfi;
    int count=0;
    for (cfi=col->begin(); cfi!=col->end();cfi++) {
      std::cout<<" Hit "<<count<<" has tof "<<cfi->timeOfFlight()<<" trackid "<<cfi->trackId() <<" bunchcr "<<cfi.bunch()<<" trigger "<<cfi.getTrigger()<<std::endl;
      count++;
     }

    // test access to EmbdSimTracks
    std::cout<<"\n=================== Starting EmbdSimTrack access ==================="<<std::endl;
    std::auto_ptr<MixCollection<EmbdSimTrack> > col2(new MixCollection<EmbdSimTrack>(cf.product()));
    MixCollection<EmbdSimTrack>::iterator cfi2;
    int count2=0;
    for (cfi2=col2->begin(); cfi2!=col2->end();cfi2++) {
      std::cout<<" EmbdSimTrack "<<count2<<" has genpart index  "<<cfi2->genpartIndex()<<" vertex Index "<<cfi2->vertIndex() <<" bunchcr "<<cfi2.bunch()<<" trigger "<<cfi2.getTrigger()<<std::endl;
      count2++; 
    }
}

