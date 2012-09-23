#include "CompatibleDetToGroupAdder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetGroupMerger.h"


using namespace std;


bool CompatibleDetToGroupAdder::add( const GeometricSearchDet& det,
				     const TrajectoryStateOnSurface& tsos, 
				     const Propagator& prop,
				     const MeasurementEstimator& est,
				     vector<DetGroup>& result) {
  if (det.hasGroups()) {
    vector<DetGroup> tmp;
    det.groupedCompatibleDetsV(tsos, prop, est,tmp);
    if (tmp.empty()) return false;
    
    if (result.empty()) result.swap(tmp);
    else                DetGroupMerger::addSameLevel(std::move(tmp), result);
  }
  else {
    vector<GeometricSearchDet::DetWithState> compatDets;
    det.compatibleDetsV( tsos, prop, est, compatDets);
    if (compatDets.empty()) return false;
    
    if (result.empty())
      result.push_back( DetGroup( 0, 1)); // empty group for insertion
    
    if (result.size() != 1)
      edm::LogError("TkDetLayers") << "CompatibleDetToGroupAdder: det is not grouped but result has more than one group!" ;
    result.front().reserve(result.front().size()+compatDets.size());
    for (vector<GeometricSearchDet::DetWithState>::const_iterator i=compatDets.begin();
	 i!=compatDets.end(); i++)
      result.front().push_back(std::move( *i));
  } 
    return true;
}

#include "TrackingTools/DetLayers/interface/GeomDetCompatibilityChecker.h"
#include "TkGeomDetCompatibilityChecker.h"

bool CompatibleDetToGroupAdder::add( const GeomDet& det,
				     const TrajectoryStateOnSurface& tsos, 
				     const Propagator& prop,
				     const MeasurementEstimator& est,
				     vector<DetGroup>& result) {
  //TkGeomDetCompatibilityChecker theCompatibilityChecker;
  GeomDetCompatibilityChecker theCompatibilityChecker;
  pair<bool, TrajectoryStateOnSurface> compat = theCompatibilityChecker.isCompatible( &det,tsos, prop, est);

  if (!compat.first) return false;

  DetGroupElement ge( &det, compat.second);

  if (result.empty())
    result.push_back( DetGroup( 0, 1)); // empty group for ge insertion

  if (result.size() != 1) 
      edm::LogError("TkDetLayers") << "CompatibleDetToGroupAdder: det is not grouped but result has more than one group!" ;
    

  result.front().push_back(ge); 
  return true;  
}




