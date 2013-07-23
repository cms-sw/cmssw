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
// $Date: 2007/04/17 21:56:54 $
// $Revision: 1.2 $
//

#include <cmath>

#include "RecoTracker/RoadMapRecord/interface/RoadMapSorting.h"

unsigned int RoadMapSorting::calculateRoadSeedIndex(const std::pair<std::vector<const Ring*>, std::vector<const Ring*> >& seed) const{
  //
  // loop over seed rings and build index: 
  // 2 rings: ring1->getindex()*1000 + ring2->getindex()
  //

  // return value
  unsigned int result = 0;

  unsigned int counter = seed.first.size() + seed.second.size();
  for ( std::vector<const Ring*>::const_iterator ring = seed.first.begin(),
	  ringEnd = seed.first.end();
	ring != ringEnd;
	++ring ) {
    result += (*ring)->getindex() * (unsigned int)std::pow((double)1000,(double)--counter);
  }
  for ( std::vector<const Ring*>::const_iterator ring = seed.second.begin(),
	  ringEnd = seed.second.end();
	ring != ringEnd;
	++ring ) {
    result += (*ring)->getindex() * (unsigned int)std::pow((double)1000,(double)--counter);
  }

  return result;
}
