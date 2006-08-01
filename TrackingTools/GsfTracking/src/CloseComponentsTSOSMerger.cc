#include "TrackingTools/GsfTracking/interface/CloseComponentsTSOSMerger.h"

#include "Geometry/Surface/interface/BoundPlane.h"
#include "TrackingTools/GsfTools/interface/BasicMultiTrajectoryState.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateAssembler.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateCombiner.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cfloat>

CloseComponentsTSOSMerger::CloseComponentsTSOSMerger (int maxNumberOfComponents,
						      const TSOSDistanceBetweenComponents* distance) :
  theMaxNumberOfComponents(maxNumberOfComponents),
  theDistance(distance->clone()) {}

TrajectoryStateOnSurface 
CloseComponentsTSOSMerger::merge(const TSOS& tsos) const {
  
  if (!tsos.isValid())  return tsos;
  
  std::vector<TSOS> unmergedComponents = tsos.components();
  int nComp = unmergedComponents.size();
  
  if (unmergedComponents.empty()) {
    edm::LogError("CloseComponentsTSOSMerger") 
      << "Trying to merge trajectory state with zero components!";
    return TSOS();
  }
  
  if (theMaxNumberOfComponents <= 0) {
    edm::LogError("CloseComponentsTSOSMerger") 
      << "Trying to merge state into zero (or less!) components, returning invalid state!";
    return TSOS();
  }
  
  if (tsos.weight() == 0) {
    edm::LogError("CloseComponentsTSOSMerger") 
      << "Trying to merge mixture with sum of weights equal to zero!";
    return TSOS();
  }
  
  if (nComp < theMaxNumberOfComponents + 1)
    return TSOS(new BasicMultiTrajectoryState(unmergedComponents));
  
  TsosMap mapUnmergedComp;
  TsosMap mapMergedComp;
  
  for (std::vector<TSOS>::const_iterator it = unmergedComponents.begin();
       it != unmergedComponents.end(); it++) {
    mapUnmergedComp.insert(std::make_pair(it->weight(), *it));
  }
    
  while (nComp > theMaxNumberOfComponents) {
    mapMergedComp.clear();
    while (nComp > theMaxNumberOfComponents && !mapUnmergedComp.empty()) {
      if (mapUnmergedComp.size() > 1) {
	std::pair<TSOS, TsosMap::iterator> pairMinDist = 
	  compWithMinDistToLargestWeight(mapUnmergedComp);
	std::vector<TSOS> comp;
	comp.push_back(mapUnmergedComp.begin()->second);
	comp.push_back(pairMinDist.first);
	mapUnmergedComp.erase(pairMinDist.second);
	mapUnmergedComp.erase(mapUnmergedComp.begin());
	TSOS mergedComp = MultiTrajectoryStateCombiner().combine(comp);
	mapMergedComp.insert(std::make_pair(mergedComp.weight(), mergedComp));
	nComp--;
      }
      else {
	mapMergedComp.insert(std::make_pair(mapUnmergedComp.begin()->first, 
					    mapUnmergedComp.begin()->second));
	mapUnmergedComp.erase(mapUnmergedComp.begin());
      }
    }
    if (mapUnmergedComp.empty() && nComp > theMaxNumberOfComponents) {
      mapUnmergedComp = mapMergedComp;
    }
  }
    
  MultiTrajectoryStateAssembler result;
    
  for (TsosMap::const_iterator it = mapUnmergedComp.begin();
       it != mapUnmergedComp.end(); it++) {
    result.addState(it->second);
  }
    
  for (TsosMap::const_iterator it = mapMergedComp.begin();
       it != mapMergedComp.end(); it++) {
    result.addState(it->second);
  }
    
  return result.combinedState();
}

std::pair< TrajectoryStateOnSurface, 
	   std::multimap<double, TrajectoryStateOnSurface>::iterator >
CloseComponentsTSOSMerger::compWithMinDistToLargestWeight(TsosMap& unmergedComp) const {
  double large = DBL_MAX;
  double minDist = large;
  TsosMap::iterator iterMinDist(0);
  for (TsosMap::iterator it = unmergedComp.begin();
       it != unmergedComp.end(); it++) {
    if (it != unmergedComp.begin()) {
      double dist = (*theDistance)(unmergedComp.begin()->second, it->second);
      if (dist < minDist) {
	iterMinDist = it;
	minDist = dist;
      }
    }
  }
  TSOS minDistComp(iterMinDist->second);
  return std::make_pair(minDistComp, iterMinDist);
}

