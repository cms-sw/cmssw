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
// $Author: gutsche $
// $Date: 2006/04/01 16:11:49 $
// $Revision: 1.4 $
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/DetId/interface/DetId.h"

class RoadSearchHelixMakerAlgorithm 
{
 public:
  
  RoadSearchHelixMakerAlgorithm(const edm::ParameterSet& conf);
  ~RoadSearchHelixMakerAlgorithm();

  /// Runs the algorithm
  void run(const TrackCandidateCollection* input,
	   const edm::EventSetup& es,
	   reco::TrackCollection &output);

 private:
  edm::ParameterSet conf_;

};

#endif
