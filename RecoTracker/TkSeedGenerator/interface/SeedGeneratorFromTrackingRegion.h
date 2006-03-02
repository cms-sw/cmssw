#ifndef SeedGeneratorFromTrackingRegion_H
#define SeedGeneratorFromTrackingRegion_H

#include "DataFormats/TrackingSeed/interface/TrackingSeed.h"
#include "DataFormats/TrackingSeed/interface/TrackingSeedCollection.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
class TrackingRegion;

/** Specializes the SeedGenerator interface for 
 *  the case where the seeds are constrained by the region.
 */

class SeedGeneratorFromTrackingRegion {
public:




 // from base class
 //  virtual SeedContainer seeds() { 
 virtual TrackingSeedCollection seeds(){
   const TrackingRegion * region = trackingRegion();
   if (region) 
     return seeds( *region); 
     else
       seeds();
   //      return SeedContainer();
 }
 
 // extend the interface for tracking region
 virtual TrackingSeedCollection seeds( const TrackingRegion& region) = 0;
     
 // if the region is not given as an argument a concrete generator must 
 // work as region provider
       
 virtual const TrackingRegion * trackingRegion() const { return 0; }
};
#endif
