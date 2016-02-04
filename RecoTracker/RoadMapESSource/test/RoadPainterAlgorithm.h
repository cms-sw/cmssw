#ifndef RoadPainterAlgorithm_h
#define RoadPainterAlgorithm_h

//
// Package:         RecoTracker/RingESSource/test
// Class:           RoadPainter
// 
// Description:     paints rings
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Dec  7 08:52:54 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/02/05 19:15:01 $
// $Revision: 1.1 $
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/RingRecord/interface/Rings.h"
#include "RecoTracker/RoadMapRecord/interface/Roads.h"

class RoadPainterAlgorithm 
{
 public:
  
  RoadPainterAlgorithm(const edm::ParameterSet& conf);
  ~RoadPainterAlgorithm();
  

  /// Runs the algorithm
  void run(const Rings* rings, const Roads *roads);

 private:

  edm::ParameterSet conf_;
  std::string pictureName_;

};

#endif
