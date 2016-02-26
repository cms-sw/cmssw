#ifndef pTFrom2Stubs_HH
#define pTFrom2Stubs_HH


#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace pTFrom2Stubs{
	float rInvFrom2(std::vector< TTTrack< Ref_Phase2TrackerDigi_> >::const_iterator trk, const StackedTrackerGeometry* theStackedGeometry);
	float pTFrom2(std::vector< TTTrack< Ref_Phase2TrackerDigi_> >::const_iterator trk, const StackedTrackerGeometry* theStackedGeometry);
}
#endif
