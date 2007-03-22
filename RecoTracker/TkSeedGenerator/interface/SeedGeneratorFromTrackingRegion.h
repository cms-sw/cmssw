#ifndef SeedGeneratorFromTrackingRegion_H
#define SeedGeneratorFromTrackingRegion_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
class TrackingRegion;

/** Specializes the SeedGenerator interface for 
 *  the case where the seeds are constrained by the region.
 */

class SeedGeneratorFromTrackingRegion {
public:

  SeedGeneratorFromTrackingRegion(const edm::ParameterSet& conf): conf_(conf)
{}
  virtual ~SeedGeneratorFromTrackingRegion(){}
  // from base class
 //  virtual SeedContainer seeds() { 
/*  virtual vector<TrajectorySeed> seeds(){ */
/*    const TrackingRegion * region = trackingRegion(); */
/*    if (region)  */
/*      return seeds( *region);  */
/*      else */
/*        seeds(); */
/*    //      return SeedContainer(); */
/*  } */
 
 // extend the interface for tracking region
/*  virtual TrackingSeedCollection seeds( const TrackingRegion& region) = 0; */
     
 // if the region is not given as an argument a concrete generator must 
 // work as region provider

   //  virtual vector<TrajectorySeed> seeds(const edm::EventSetup& c,const TrackingRegion& region)=0;
   virtual  void  seeds(TrajectorySeedCollection &output,
			const edm::Event& ev,
			const edm::EventSetup& c,
			const TrackingRegion& region){};
  virtual const TrackingRegion * trackingRegion() const { return 0; }

  const edm::ParameterSet& pSet(){return conf_;}

 private:
  const edm::ParameterSet conf_;

};
#endif
