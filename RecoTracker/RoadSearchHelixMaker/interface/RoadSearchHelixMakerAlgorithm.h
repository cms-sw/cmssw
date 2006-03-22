#ifndef HelixMakerAlgorithm_h
#define HelixMakerAlgorithm_h

//
// Package:         RecoTracker/RoadSearchHelixMaker
// Class:           RoadSearchHelixMakerAlgorithm
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

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"

#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

class RoadSearchHelixMakerAlgorithm 
{
 public:
  
  RoadSearchHelixMakerAlgorithm(const edm::ParameterSet& conf);
  ~RoadSearchHelixMakerAlgorithm();

  /// Runs the algorithm
  void run(const RoadSearchCloudCollection* input,
	   const edm::EventSetup& es,
	   reco::TrackCollection &output);

  bool isBarrelSensor(DetId id);

 private:
  edm::ParameterSet conf_;

};

#endif
