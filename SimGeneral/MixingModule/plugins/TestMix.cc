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
// $Id: TestMix.cc,v 1.1 2007/04/19 14:51:54 uberthon Exp $
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
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "TestMix.h"

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

  std::auto_ptr<MixCollection<PCaloHit> > colcalo(new MixCollection<PCaloHit>(cf.product(),"EcalHitsEB",std::pair<int,int>(0,0) ));

 std::cout << " Number of fired cells " << colcalo->size() << std::endl;

}

