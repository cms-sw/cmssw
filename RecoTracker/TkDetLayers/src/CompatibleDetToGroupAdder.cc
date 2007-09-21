#include "RecoTracker/TkDetLayers/interface/CompatibleDetToGroupAdder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"


using namespace std;


bool CompatibleDetToGroupAdder::add( const GeometricSearchDet& det,
				     const TrajectoryStateOnSurface& tsos, 
				     const Propagator& prop,
				     const MeasurementEstimator& est,
				     vector<DetGroup>& result) const
{
  if (det.hasGroups()) {
    vector<DetGroup> tmp( det.groupedCompatibleDets( tsos, prop, est));

    if (!tmp.empty()) {
      if (result.empty()) result = tmp;
      else                DetGroupMerger().addSameLevel( tmp, result);
      return true;
    }
  }
  else {
    vector<GeometricSearchDet::DetWithState> compatDets = det.compatibleDets( tsos, prop, est);
    if (!compatDets.empty()) {
	if (result.empty()) {
	  result.push_back( DetGroup( 0, 1)); // empty group for insertion
	}
	if (result.size() != 1) {
	  edm::LogError("TkDetLayers") << "CompatibleDetToGroupAdder: det is not grouped but result has more than one group!" ;
	}
	for (vector<GeometricSearchDet::DetWithState>::const_iterator i=compatDets.begin();
	     i!=compatDets.end(); i++) {
	  result.front().push_back( *i);
	}
	return true;
    }
  }

  return false;
}

#include "TrackingTools/DetLayers/interface/GeomDetCompatibilityChecker.h"
#include "RecoTracker/TkDetLayers/interface/TkGeomDetCompatibilityChecker.h"

bool CompatibleDetToGroupAdder::add( const GeomDet& det,
				     const TrajectoryStateOnSurface& tsos, 
				     const Propagator& prop,
				     const MeasurementEstimator& est,
				     vector<DetGroup>& result) const
{
  //TkGeomDetCompatibilityChecker theCompatibilityChecker;
  GeomDetCompatibilityChecker theCompatibilityChecker;
  pair<bool, TrajectoryStateOnSurface> compat = theCompatibilityChecker.isCompatible( &det,tsos, prop, est);

  if (compat.first) {
    DetGroupElement ge( &det, compat.second);
    if (result.empty()) {
      result.push_back( DetGroup( 0, 1)); // empty group for ge insertion
    }
    else {
      if (result.size() != 1) {
	edm::LogError("TkDetLayers") << "CompatibleDetToGroupAdder: det is not grouped but result has more than one group!" ;
      }
    }
    result.front().push_back(ge); 
    return true;
  }
  return false;
}




