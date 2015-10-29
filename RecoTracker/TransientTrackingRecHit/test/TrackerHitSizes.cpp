#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit1D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
//
// For FAMOS
//
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"  
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"                         
#include "DataFormats/TrackerRecHit2D/interface/FastMatchedTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastProjectedTrackerRecHit.h"

#include<iostream>

#define PSIZE(CNAME) std::cout << #CNAME << ": " << sizeof(CNAME) << std::endl

int main() {
  std::cout << "sizes" << std::endl;
  PSIZE(SiPixelRecHit);
  PSIZE(SiStripRecHit1D);
  PSIZE(SiStripRecHit2D);
  PSIZE(SiStripMatchedRecHit2D);
  PSIZE(ProjectedSiStripRecHit2D);

  std::cout << std::endl;

  PSIZE(GenericTransientTrackingRecHit);

  PSIZE(FastTrackerRecHit);
  PSIZE(FastSingleTrackerRecHit);
  PSIZE(FastMatchedTrackerRecHit);
  PSIZE(FastProjectedTrackerRecHit);



  return 0;
}
