// -*- C++ -*-
//
// Package:    RoadMapTest
// Class:      RoadMapTest
// 
/**\class RoadMapTest RoadMapTest.cc RoadMap/test/RoadMapTest.cc

 Description: test RoadMap by reading the ascii file and dumping it to a new file

*/
//
// Original Author:  Oliver Gutsche
//         Created:  Sun Nov 20 16:30:00 CEST 2005
// $Id: RoadMapTest.cc,v 1.2 2006/03/23 01:52:10 gutsche Exp $
//
//


// system include files

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/RoadMapRecord/interface/Roads.h"

//
//
// class decleration
//

class RoadMapTest : public edm::EDAnalyzer {
   public:
      explicit RoadMapTest( const edm::ParameterSet& );
      ~RoadMapTest();

      virtual void analyze( const edm::Event&, const edm::EventSetup& );
   private:
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RoadMapTest::RoadMapTest( const edm::ParameterSet& iConfig )
{
   //now do what ever initialization is needed

}


RoadMapTest::~RoadMapTest()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
RoadMapTest::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  edm::ESHandle<Roads> roads;
  iSetup.get<TrackerDigiGeometryRecord>().get(roads);
  roads->dump("roads.dat");

}

//define this as a plug-in
DEFINE_FWK_MODULE(RoadMapTest);
