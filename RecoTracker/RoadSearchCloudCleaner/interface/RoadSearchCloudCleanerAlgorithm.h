#ifndef CloudCleanerAlgorithm_h
#define CloudCleanerAlgorithm_h

//
// Package:         RecoTracker/RoadSearchCloudCleaner
// Class:           RoadSearchCloudCleanerAlgorithm
// 
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Feb 19 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/03/28 23:12:10 $
// $Revision: 1.1 $
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"

class RoadSearchCloudCleanerAlgorithm 
{
 public:
  
  RoadSearchCloudCleanerAlgorithm(const edm::ParameterSet& conf);
  ~RoadSearchCloudCleanerAlgorithm();

  /// Runs the algorithm
  void run(const RoadSearchCloudCollection* input,
	   const edm::EventSetup& es,
	   RoadSearchCloudCollection &output);

 private:
  double mergingFraction_;
  unsigned int maxRecHitsInCloud_;

};

#endif
