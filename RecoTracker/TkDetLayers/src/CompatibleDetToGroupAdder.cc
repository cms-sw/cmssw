#include "RecoTracker/TkDetLayers/interface/CompatibleDetToGroupAdder.h"
//#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"




bool CompatibleDetToGroupAdder::add( const GeometricSearchDet& det,
				     const TrajectoryStateOnSurface& tsos, 
				     const Propagator& prop,
				     const MeasurementEstimator& est,
				     vector<DetGroup>& result) const
{
  cout << "Dummy implementation of CompatibleDetToGroupAdder::add() " << endl; 
  return false;
}

bool CompatibleDetToGroupAdder::add( const GeomDet& det,
				     const TrajectoryStateOnSurface& tsos, 
				     const Propagator& prop,
				     const MeasurementEstimator& est,
				     vector<DetGroup>& result) const
{
  cout << "Dummy implementation of CompatibleDetToGroupAdder::add() " << endl; 
  return false;
}



/*
bool CompatibleDetToGroupAdder::add( const Det& det,
				     const TrajectoryStateOnSurface& tsos, 
				     const Propagator& prop,
				     const MeasurementEstimator& est,
				     vector<DetGroup>& result) const
{
//   cout << "CompatibleDetToGroupAdder called for " 
//        << typeid(det).name() << " at pos (" 
//        << det.position().perp() << ", " 
//        << det.position().phi()  << ", " 
//        << det.position().z()    << ") " 
//        << endl;

  if (det.hasGroups()) {
    const CompositeDet& cdet = dynamic_cast< const CompositeDet&>(det);
    vector<DetGroup> tmp( cdet.groupedCompatibleDets( tsos, prop, est));

//     cout << "CompatibleDetToGroupAdder: det hasGroups, returned group vector of size "
// 	 << tmp.size() << endl; 

    if (!tmp.empty()) {
      if (result.empty()) result = tmp;
      else                DetGroupMerger().addSameLevel( tmp, result);
      return true;
    }
  }
  else {
    const CompositeDet* cdetp = dynamic_cast< const CompositeDet*>(&det);
    if (cdetp != 0) {
      vector<CompositeDet::DetWithState> compatDets = cdetp->compatibleDets( tsos, prop, est);
      if (!compatDets.empty()) {
	if (result.empty()) {
	  result.push_back( DetGroup( 0, 1)); // empty group for insertion
	}
	if (result.size() != 1) {
	  cout << "CompatibleDetToGroupAdder: det is not grouped but result has more than one group!" 
	       << endl;
	}
	for (vector<CompositeDet::DetWithState>::const_iterator i=compatDets.begin();
	     i!=compatDets.end(); i++) {
	  result.front().push_back( *i);
	}
	return true;
      }
    }
    else {

      pair<bool, TrajectoryStateOnSurface> compat = det.compatible( tsos, prop, est);

//       cout << "CompatibleDetToGroupAdder: det has no groups, is compatible? " << compat.first << endl;

      if (compat.first) {
	DetGroupElement ge( &det, compat.second);
	if (result.empty()) {
	  result.push_back( DetGroup( 0, 1)); // empty group for ge insertion
	}
	else {
	  if (result.size() != 1) {
	    cout << "CompatibleDetToGroupAdder: det is not grouped but result has more than one group!" 
		 << endl;
	  }
	}
	result.front().push_back(ge); 
	return true;
      }
    }
  }

//   cout << "CompatibleDetToGroupAdder: returning false" << endl;
  return false;
}
*/
