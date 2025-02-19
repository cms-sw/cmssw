//
// Package:         RecoTracker/RoadSearchHelixMaker/test
// Class:           RoadSearchTrackReaderAlgorithm
// 
// Description:     read and analyze TrackCandidates from the RoadSearch
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 29 20:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/06/20 09:09:19 $
// $Revision: 1.1 $
//

#ifndef RoadSearchTrackReaderAlgorithm_h
#define RoadSearchTrackReaderAlgorithm_h

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

class RoadSearchTrackReaderAlgorithm 
{
 public:
  
  RoadSearchTrackReaderAlgorithm(const edm::ParameterSet& conf);
  ~RoadSearchTrackReaderAlgorithm();
  

  /// Runs the algorithm
  void run(const reco::TrackCollection* input);

 private:

  edm::ParameterSet conf_;
};

#endif
