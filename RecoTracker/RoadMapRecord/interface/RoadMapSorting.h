#ifndef RoadMapRecord_RoadMapSorting_H
#define RoadMapRecord_RoadMapSorting_H

//
// Package:         RecoTracker/RoadMapRecord
// Class:           RoadMapSorting
// 
// Description:     defines sorting of Roads keys (RoadSeed)
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Tue Apr 17 20:31:32 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/04/17 21:56:53 $
// $Revision: 1.2 $
//

#include <vector>
#include <utility>
#include "RecoTracker/RingRecord/interface/Ring.h"

class RoadMapSorting {
public:

  RoadMapSorting() {}
  
  
  bool operator()(const std::pair<std::vector<const Ring*>, std::vector<const Ring*> >& a, const std::pair<std::vector<const Ring*>, std::vector<const Ring*> >& b) const {
    return calculateRoadSeedIndex(a) < calculateRoadSeedIndex(b);
  }
  
 private:

  unsigned int calculateRoadSeedIndex(const std::pair<std::vector<const Ring*>, std::vector<const Ring*> >& seed) const;
    
};
#endif
