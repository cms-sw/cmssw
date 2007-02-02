#ifndef GlobalTrackingRegion_H
#define GlobalTrackingRegion_H

/** \class GlobalTrackingRegion
 * An implementation of the TrackingRegion where the region of interest is
 * global, ie there are no constraints on the allowed direction of particles
 * of interest
 */

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include <vector>

class GlobalTrackingRegion :  public TrackingRegionBase {
public:

  /** Construct from minimal track P_t, and origin size and position.
   *  The origin region is a cylinder of radius originRadius, half length 
   *  originHalfLength, positioned at (0,0,originZPos).
   *  This class does not provide the pssibility to displace the origin
   *  in the transverse plane. 
   */
  GlobalTrackingRegion ( float ptMin = 1., float originRadius = 0.2, 
      float originHalfLength = 15., float originZPos = 0.,
      bool precise = false)
    : TrackingRegionBase(GlobalVector( 0, 0, 0), GlobalPoint( 0, 0, originZPos),
      Range( -1/ptMin, 1/ptMin), originRadius, originHalfLength),
      thePrecise(precise) { }

 
/*   virtual HitRZCompatibility* checkRZ( */
/* 				      const DetLayer* layer, SiPixelRecHit outerHit) const; */
  virtual HitRZCompatibility * checkRZ(const DetLayer* layer,  
				       const TrackingRecHit*  outerHit,
				       const edm::EventSetup& iSetup) const;
  virtual GlobalTrackingRegion* clone() const { 
    return new GlobalTrackingRegion(*this);
  }

  static std::string const & name() 
    { static std::string local("GlobalTrackingRegion"); return local; }
  virtual std::string const & getName() const {return name();}


private:
  bool  thePrecise;
};
#endif
