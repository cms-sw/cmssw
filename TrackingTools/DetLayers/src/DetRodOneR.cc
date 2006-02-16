#include "TrackingTools/DetLayers/interface/DetRodOneR.h"
#include "TrackingTools/DetLayers/interface/RodPlaneBuilderFromDet.h"
#include "TrackingTools/DetLayers/interface/GSDUnit.h"

#include <algorithm>
#include <cmath>


DetRodOneR::~DetRodOneR(){}

DetRodOneR::DetRodOneR(vector<Det*>::const_iterator first,
		       vector<Det*>::const_iterator last)
  : theDets(first,last)
{
  initialize();
}

DetRodOneR::DetRodOneR( const vector<Det*>& dets)
  : theDets(dets) 
{
  initialize();
}


DetRodOneR::DetRodOneR( const vector<GSDUnit*>& detUnits){
  theDets.reserve(detUnits.size());
  for( vector<GSDUnit*>::const_iterator i=detUnits.begin();
       i != detUnits.end(); i++)     theDets.push_back(*i);
  initialize();
}

void DetRodOneR::initialize()
{
  // assume the dets ARE in a rod and Z ordered
  //sort( theDets.begin(), theDets.end(), DetLessZ());
  
  setPlane( RodPlaneBuilderFromDet()( theDets));
  
}


bool DetRodOneR::add( int idet, vector<DetWithState>& result,
		      const TrajectoryStateOnSurface& startingState,
		      const Propagator& prop, 
		      const MeasurementEstimator& est) const
{
  pair<bool,TrajectoryStateOnSurface> compat =
    theDets[idet]->compatible( startingState, prop, est);

  if (compat.first) {
    result.push_back( DetWithState( theDets[idet], compat.second));
  }

  return compat.first;
}

