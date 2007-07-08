//
// Package:         RecoTracker/RoadSearchCloudMaker/test
// Class:           RoadSearchCloudDumper.cc
// 
// Description:     Hit Dumper
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Mon Feb  5 21:24:36 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/03/07 21:46:50 $
// $Revision: 1.3 $
//


#include <sstream>

#include "RecoTracker/RoadSearchCloudMaker/test/RoadSearchCloudDumper.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/RingRecord/interface/RingRecord.h"
#include "RecoTracker/RingRecord/interface/Rings.h"

#include "TrackingTools/RoadSearchHitAccess/interface/RoadSearchDetIdHelper.h"


RoadSearchCloudDumper::RoadSearchCloudDumper(const edm::ParameterSet& conf) {

  // retrieve InputTags for rechits
  roadSearchCloudsInputTag_ = conf.getParameter<edm::InputTag>("RoadSearchCloudInputTag");

  ringsLabel_ = conf.getParameter<std::string>("RingsLabel");

}

RoadSearchCloudDumper::~RoadSearchCloudDumper(){
}

void RoadSearchCloudDumper::analyze(const edm::Event& e, const edm::EventSetup& es){

    
  // Step A: Get Inputs 
  edm::Handle<RoadSearchCloudCollection> cloudHandle;
  e.getByLabel(roadSearchCloudsInputTag_, cloudHandle);
  const RoadSearchCloudCollection *clouds = cloudHandle.product();

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
	 << "Cloud: Cloud number" << std::endl
	 << std::endl 
	 << "Hit: Hit_number rawid ringindex global_r global_phi global_x global_y global_z " << std::endl
	 << std::endl;

  output << std::endl
	 << "cloud collection size " << clouds->size() << std::endl
	 << std::endl;

  unsigned int ncloud=0;
  for ( RoadSearchCloudCollection::const_iterator cloud = clouds->begin(), cloudsEnd = clouds->end();
	cloud != cloudsEnd; 
	++cloud ) {

    output << "Cloud: " << ++ncloud << std::endl;
    
    unsigned int nhit = 0;
    for ( RoadSearchCloud::RecHitVector::const_iterator hit = cloud->begin_hits() , hitEnd = cloud->end_hits();
	  hit != hitEnd;
	  ++hit ) {
      DetId id = (*hit)->geographicalId();
      GlobalPoint outer = tracker->idToDet(id)->surface().toGlobal((*hit)->localPosition());
      const Ring *ring = rings->getRing(RoadSearchDetIdHelper::ReturnRPhiId(id));
      output<< "Hit: " << ++nhit << " " << id.rawId() << " " << ring->getindex() << " "
	    << outer.perp() << " " << outer.phi() 
	    << " " << outer.x() << " " << outer.y() << " " << outer.z() << std::endl;
    } // end of loop over cloud hits
  } // end of loop over clouds
  
  edm::LogInfo("RoadSearchCloudDumper") << output.str();

}
