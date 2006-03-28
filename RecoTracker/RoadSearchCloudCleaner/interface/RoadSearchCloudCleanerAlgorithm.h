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
// $Author: stevew $
// $Date: 2006/02/22 01:16:14 $
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
  edm::ParameterSet conf_;

};

#endif
