#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.h"

//#include "Vertex/GaussianSumVertexFit/interface/KeepingNonZeroWeightsMerger.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateCombiner.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateAssembler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cfloat>

CloseComponentsMerger::CloseComponentsMerger (int maxNumberOfComponents,
					      const DistanceBetweenComponents* distance) :
  theMaxNumberOfComponents(maxNumberOfComponents),
  theDistance(distance->clone()) {}
                                                                                                               
RCMultiGaussianState 
CloseComponentsMerger::merge(const RCMultiGaussianState& mgs) const
{
// ThS: Can not check for TSOS invalidity
//   if (!tsos.isValid()) {
//     return tsos;
//   } else {
    
    //TSOS trimmedTsos = KeepingNonZeroWeightsMerger().merge(tsos);

  SGSVector unmergedComponents = mgs->components();
  SGSVector finalComponents;
  int nComp = unmergedComponents.size();
    
  if (unmergedComponents.empty()) {
    edm::LogError("CloseComponentsMerger") 
      << "Trying to merge trajectory state with zero components!";
    return mgs; // ThS: TSOS();
  }

// ThS: Don't you want to throw an exception at construction of the class?
  if (theMaxNumberOfComponents <= 0) {
    edm::LogError("CloseComponentsMerger") 
      << "Trying to merge state into zero (or less!) components, returning invalid state!";
    return mgs; // ThS: TSOS();
  }
    
// ThS: Of course, here the TSOS will not be invalid. But it will have 0 components
  if (mgs->weight() == 0) {
    edm::LogError("CloseComponentsMerger") 
      << "Trying to merge mixture with sum of weights equal to zero!";
    return mgs->createNewState(finalComponents);
  }
    
  if (nComp < theMaxNumberOfComponents + 1) return mgs;
// ThS: Why not the initial object, as above?
//      return TSOS(new BasicMultiTrajectoryState(unmergedComponents));
    
  SingleStateMap mapUnmergedComp;
  SingleStateMap mapMergedComp;

  for (SGSVector::const_iterator it = unmergedComponents.begin();
       it != unmergedComponents.end(); it++) {
    mapUnmergedComp.insert(std::make_pair((**it).weight(), *it));
  }

  while (nComp > theMaxNumberOfComponents) {
    mapMergedComp.clear();
    while (nComp > theMaxNumberOfComponents && !mapUnmergedComp.empty()) {
      if (mapUnmergedComp.size() > 1) {
	std::pair<SGS, SingleStateMap::iterator> pairMinDist = 
	  compWithMinDistToLargestWeight(mapUnmergedComp);
	SGSVector comp;
	comp.push_back(mapUnmergedComp.begin()->second);
	comp.push_back(pairMinDist.first);
	mapUnmergedComp.erase(pairMinDist.second);
	mapUnmergedComp.erase(mapUnmergedComp.begin());
	SGS mergedComp = MultiGaussianStateCombiner().combine(comp);
	mapMergedComp.insert(std::make_pair(mergedComp->weight(), mergedComp));
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

  MultiGaussianStateAssembler result(mgs);

  for (SingleStateMap::const_iterator it = mapUnmergedComp.begin();
       it != mapUnmergedComp.end(); it++) {
    result.addState(it->second);
  }

  for (SingleStateMap::const_iterator it = mapMergedComp.begin();
       it != mapMergedComp.end(); it++) {
    result.addState(it->second);
  }
    
    
  return result.combinedState();
}

std::pair<RCSingleGaussianState, 
	  std::multimap<double, RCSingleGaussianState>::iterator>
CloseComponentsMerger::compWithMinDistToLargestWeight(SingleStateMap& unmergedComp) const {
  double large = DBL_MAX;
  double minDist = large;
  SingleStateMap::iterator iterMinDist(0);
  for (SingleStateMap::iterator it = unmergedComp.begin();
       it != unmergedComp.end(); it++) {
    if (it != unmergedComp.begin()) {
      double dist = (*theDistance)(unmergedComp.begin()->second, it->second);
      if (dist < minDist) {
	iterMinDist = it;
	minDist = dist;
      }
    }
  }
  SGS minDistComp(iterMinDist->second);
  return std::make_pair(minDistComp, iterMinDist);
}

