#ifndef CloudCleanerAlgorithm_h
#define CloudCleanerAlgorithm_h

//
// Package:         RecoTracker/RoadSearchCloudMaker
// Class:           RoadSearchCloudCleanerAlgorithm
// 
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Feb 19 22:00:00 UTC 2006
//
// $Author: stevew $
// $Date: 2006/02/10 22:54:52 $
// $Revision: 1.2 $
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/TrackingSeed/interface/TrackingSeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"

#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

class RoadSearchCloudCleanerAlgorithm 
{
 public:
  
  RoadSearchCloudCleanerAlgorithm(const edm::ParameterSet& conf);
  ~RoadSearchCloudCleanerAlgorithm();

  /// Runs the algorithm
  void run(const RoadSearchCloudCollection* input,
           const TrackingSeedCollection* seeds,
	   const edm::EventSetup& es,
	   RoadSearchCloudCollection &output);

 private:
  edm::ParameterSet conf_;

};

#endif
