#ifndef TransientTrackingRecHit_H
#define TransientTrackingRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Surface.h" 

#include "FWCore/Utilities/interface/GCC11Compatibility.h"


#ifdef COUNT_HITS
void countTTRH( TrackingRecHit::Type);
#else
inline void countTTRH( TrackingRecHit::Type){}
#endif

typedef TrackingRecHit TransientTrackingRecHit;

#endif

