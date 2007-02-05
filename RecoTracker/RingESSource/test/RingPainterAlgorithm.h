#ifndef RingPainterAlgorithm_h
#define RingPainterAlgorithm_h

//
// Package:         RecoTracker/RingESSource/test
// Class:           RingPainter
// 
// Description:     paints rings
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Dec  7 08:52:54 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/06/20 09:09:19 $
// $Revision: 1.1 $
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/RingRecord/interface/Rings.h"

class RingPainterAlgorithm 
{
 public:
  
  RingPainterAlgorithm(const edm::ParameterSet& conf);
  ~RingPainterAlgorithm();
  

  /// Runs the algorithm
  void run(const Rings* rings);

 private:

  edm::ParameterSet conf_;
  std::string pictureName_;

};

#endif
