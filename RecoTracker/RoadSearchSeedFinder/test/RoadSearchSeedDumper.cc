//
// Package:         RecoTracker/RoadSearchSeedFinder/test
// Class:           RoadSearchSeedDumper.cc
// 
// Description:     Seed Dumper
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Mon Feb  5 21:24:36 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/03/07 21:46:51 $
// $Revision: 1.3 $
//


#include <sstream>

#include "RecoTracker/RoadSearchSeedFinder/test/RoadSearchSeedDumper.h"

#include "DataFormats/RoadSearchSeed/interface/RoadSearchSeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/RingRecord/interface/RingRecord.h"
#include "RecoTracker/RingRecord/interface/Rings.h"

#include "TrackingTools/RoadSearchHitAccess/interface/RoadSearchDetIdHelper.h"


RoadSearchSeedDumper::RoadSearchSeedDumper(const edm::ParameterSet& conf) {

  // retrieve InputTags for rechits
  roadSearchSeedsInputTag_ = conf.getParameter<edm::InputTag>("RoadSearchSeedInputTag");

  ringsLabel_ = conf.getParameter<std::string>("RingsLabel");

}

RoadSearchSeedDumper::~RoadSearchSeedDumper(){
}

void RoadSearchSeedDumper::analyze(const edm::Event& e, const edm::EventSetup& es){

    
  // Step A: Get Inputs 
  edm::Handle<RoadSearchSeedCollection> seedHandle;
  e.getByLabel(roadSearchSeedsInputTag_, seedHandle);
  const RoadSearchSeedCollection *seeds = seedHandle.product();

  // get tracker geometry
  edm::ESHandle<TrackerGeometry> trackerHandle;
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  const TrackerGeometry *tracker = trackerHandle.product();

  // get rings
  edm::ESHandle<Rings> ringsHandle;
  es.get<RingRecord>().get(ringsLabel_, ringsHandle);
  const Rings *rings = ringsHandle.product();

  std::ostringstream output;

  output << std::endl 
	 << "Format: " << std::endl 
	 << std::endl 
	 << "Seed: Seed number" << std::endl
	 << std::endl 
	 << "Hit: Hit_number rawid ringindex global_r global_phi global_x global_y global_z " << std::endl
	 << std::endl;


  output << std::endl
	 << "seed collection size " << seeds->size() << std::endl
	 << std::endl;

  unsigned int nseed=0;
  for ( RoadSearchSeedCollection::const_iterator seed = seeds->begin(), seedsEnd = seeds->end();
	seed != seedsEnd; 
	++seed ) {

    output << "Seed: " << ++nseed << std::endl;
    
    unsigned int nhit = 0;
    
    for (RoadSearchSeed::HitVector::const_iterator hit = seed->begin() ; hit != seed->end();
	  ++hit ) {
      DetId id = (*hit)->geographicalId();
      GlobalPoint outer = tracker->idToDet(id)->surface().toGlobal((*hit)->localPosition());
      const Ring *ring = rings->getRing(RoadSearchDetIdHelper::ReturnRPhiId(id));
      output<< "Hit: " << ++nhit << " " << id.rawId() << " " << ring->getindex() << " "
	    << outer.perp() << " " << outer.phi() 
	    << " " << outer.x() << " " << outer.y() << " " << outer.z() << std::endl;
    } // end of loop over seed hits
  } // end of loop over seeds

  edm::LogInfo("RoadSearchSeedDumper") << output.str();

}
