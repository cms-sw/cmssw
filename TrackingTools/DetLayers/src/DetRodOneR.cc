#include "TrackingTools/DetLayers/interface/DetRodOneR.h"
#include "TrackingTools/DetLayers/interface/RodPlaneBuilderFromDet.h"

// #include <Utilities/General/interface/precomputed_value_sort.h>
// #include <Geometry/CommonDetUnit/interface/DetSorting.h>

#include <algorithm>
#include <cmath>


DetRodOneR::~DetRodOneR(){}

DetRodOneR::DetRodOneR(vector<const GeomDet*>::const_iterator first,
		       vector<const GeomDet*>::const_iterator last)
  : theDets(first,last)
{
  initialize();
}

DetRodOneR::DetRodOneR( const vector<const GeomDet*>& dets)
  : theDets(dets) 
{
  initialize();
}


void DetRodOneR::initialize()
{
  // assume the dets ARE in a rod AND Z ordered
  // precomputed_value_sort( theDets.begin(), theDets.end(), geomsort::DetZ());
  
  setPlane( RodPlaneBuilderFromDet()( theDets));
  
}


// It needs that the basic component to have the compatible() method
bool DetRodOneR::add( int idet, vector<DetWithState>& result,
		      const TrajectoryStateOnSurface& startingState,
		      const Propagator& prop, 
		      const MeasurementEstimator& est) const
{
  pair<bool,TrajectoryStateOnSurface> compat = 
    theCompatibilityChecker.isCompatible(theDets[idet],startingState, prop, est);
  
  if (compat.first) {
    result.push_back( DetWithState( theDets[idet], compat.second));
  }

  return compat.first;
}
