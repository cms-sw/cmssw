#include "TrackingTools/DetLayers/interface/ForwardDetRingOneZ.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h" 
#include "TrackingTools/DetLayers/interface/ForwardRingDiskBuilderFromDet.h"
//#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/DetSorting.h>

#include <algorithm>
#include <cmath>

using namespace std;

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
  // assume the dets ARE in a ring
  // sort them in phi
  precomputed_value_sort( theDets.begin(), theDets.end(), geomsort::DetPhi());
  setDisk(ForwardRingDiskBuilderFromDet()(theDets));
}


bool ForwardDetRingOneZ::add(int idet, vector<DetWithState>& result,
			     const TrajectoryStateOnSurface& tsos,
			     const Propagator& prop, 
			     const MeasurementEstimator& est) const {
  pair<bool,TrajectoryStateOnSurface> compat =
    theCompatibilityChecker.isCompatible(theDets[idet],tsos, prop, est);

  if (compat.first) {
    result.push_back(DetWithState(theDets[idet], compat.second));
  }

  return compat.first;
}

