#include "TrackingTools/DetLayers/interface/ForwardDetRingOneZ.h"
//#include "CommonDet/DetLayout/src/DetLessPhi.h"
#include "TrackingTools/DetLayers/interface/ForwardRingDiskBuilderFromDet.h"
//#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include <algorithm>
#include <cmath>


ForwardDetRingOneZ::~ForwardDetRingOneZ(){}

ForwardDetRingOneZ::ForwardDetRingOneZ(vector<const GeomDet*>::const_iterator first,
				       vector<const GeomDet*>::const_iterator last)
  : theDets(first,last)
{
  initialize();
}

ForwardDetRingOneZ::ForwardDetRingOneZ( const vector<const GeomDet*>& dets)
  : theDets(dets)
{
  initialize();
}


void ForwardDetRingOneZ::initialize()
{
  // assume the dets ARE in a ring AND phi ordered
  //sort( theDets.begin(), theDets.end(), DetLessPhi());

  setDisk( ForwardRingDiskBuilderFromDet()( theDets));

}

/*
bool ForwardDetRingOneZ::add( int idet, vector<DetWithState>& result,
			  const FreeTrajectoryState& fts,
			  const Propagator& prop, 
			  const MeasurementEstimator& est) const
{
  pair<bool,TrajectoryStateOnSurface> compat =
    theDets[idet]->compatible( fts, prop, est);

  if (compat.first) {
    result.push_back( DetWithState( theDets[idet], compat.second));
  }

  return compat.first;
}
*/
