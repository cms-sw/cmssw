//
// Package:         RecoTracker/RoadMapRecord
// Class:           RoadMapTest
// 
// Description:     test roads
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sun Feb  4 19:15:56 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/03/01 08:05:00 $
// $Revision: 1.5 $
//

#include "RecoTracker/RoadMapRecord/test/RoadMapTest.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/RoadMapRecord/interface/RoadMapRecord.h"
#include "RecoTracker/RoadMapRecord/interface/Roads.h"

RoadMapTest::RoadMapTest( const edm::ParameterSet& iConfig )
{

   dumpRoads_ = iConfig.getUntrackedParameter<bool>("DumpRoads");
   fileName_  = iConfig.getUntrackedParameter<std::string>("FileName");
   roadLabel_  = iConfig.getUntrackedParameter<std::string>("RoadLabel");
}


RoadMapTest::~RoadMapTest()
{
 
}


void
RoadMapTest::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  edm::ESHandle<Roads> roads;
  iSetup.get<RoadMapRecord>().get(roadLabel_,roads);
  if ( dumpRoads_ ) {
    roads->dump(fileName_);
  }

}
