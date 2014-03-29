#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit1D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
//
// For FAMOS
//
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"  
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"                         
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"

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

  PSIZE(TSiStripRecHit2DLocalPos);
  PSIZE(TSiStripRecHit1D);
  PSIZE(TSiStripMatchedRecHit);
  PSIZE(TSiPixelRecHit);
  PSIZE(InvalidTransientRecHit);
  PSIZE(ProjectedRecHit2D);
  PSIZE(GenericTransientTrackingRecHit);

  PSIZE(SiTrackerGSRecHit2D);
  PSIZE(SiTrackerGSMatchedRecHit2D);



  return 0;
}
