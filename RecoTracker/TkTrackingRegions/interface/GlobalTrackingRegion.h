#ifndef GlobalTrackingRegion_H
#define GlobalTrackingRegion_H

/** \class GlobalTrackingRegion
 * An implementation of the TrackingRegion where the region of interest is
 * global, ie there are no constraints on the allowed direction of particles
 * of interest
 */

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include <vector>

class GlobalTrackingRegion GCC11_FINAL :  public TrackingRegion {
public:

  /** Construct from minimal track P_t, and origin size and position.
   *  The origin region is a cylinder of radius originRadius, half length 
   *  originHalfLength, positioned at "origin".
   *  This class DOES provide the possibility to displace the origin
   *  in the transverse plane. 
   */
  GlobalTrackingRegion ( float ptMin, const GlobalPoint & origin, 
      float originRadius, float originHalfLength, bool precise=false)
    :  TrackingRegionBase(GlobalVector( 0, 0, 0), origin,
      Range( -1/ptMin, 1/ptMin), originRadius, originHalfLength),
      thePrecise(precise) { }

  // obsolete constructor
  GlobalTrackingRegion ( float ptMin = 1., float originRadius = 0.2, 
      float originHalfLength = 22.7, float originZPos = 0.,
      bool precise = false)
    : TrackingRegionBase(GlobalVector( 0, 0, 0), GlobalPoint( 0, 0, originZPos),
      Range( -1/ptMin, 1/ptMin), originRadius, originHalfLength),
      thePrecise(precise) { }

  TrackingRegion::Hits hits(
      const edm::Event& ev,
      const edm::EventSetup& es,
      const SeedingLayerSetsHits::SeedingLayer& layer) const override;

 
  virtual HitRZCompatibility * checkRZ(const DetLayer* layer,  
				       const Hit &  outerHit,
				       const edm::EventSetup& iSetup,
				       const DetLayer* outerlayer=0,
				       float lr=0, float gz=0, float dr=0, float dz=0) const ;

  virtual GlobalTrackingRegion* clone() const { 
    return new GlobalTrackingRegion(*this);
  }

  virtual std::string name() const { return "GlobalTrackingRegion"; }
  virtual std::string print() const;

private:
  bool  thePrecise;
};
#endif
