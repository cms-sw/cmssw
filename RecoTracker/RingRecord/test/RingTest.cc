//
// Package:         RecoTracker/RingMakerESProducer/test
// Class:           RingTest
// 
// Description:     test rings
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Fri Dec  8 10:15:02 UTC 2006
//
// $Author: noeding $
// $Date: 2006/08/12 00:23:51 $
// $Revision: 1.6 $
//

#include "RecoTracker/RingRecord/test/RingTest.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/RingRecord/interface/RingRecord.h"
#include "RecoTracker/RingRecord/interface/Rings.h"

RingTest::RingTest( const edm::ParameterSet& iConfig )
{

   dumpRings_ = iConfig.getUntrackedParameter<bool>("DumpRings");
   fileName_  = iConfig.getUntrackedParameter<std::string>("FileName");

}

RingTest::~RingTest()
{
 
}

void
RingTest::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  edm::ESHandle<Rings> rings;
  iSetup.get<RingRecord>().get(rings);
  if ( dumpRings_ ) {
    rings->dump(fileName_);
  }

}
