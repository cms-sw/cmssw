#ifndef StubPtConsistency_HH
#define StubPtConsistency_HH

#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace StubPtConsistency {
  float getConsistency(TTTrack< Ref_PixelDigi_ > aTrack, const StackedTrackerGeometry* theStackedGeometry, double mMagneticFieldStrength, int nPar);
}
#endif
