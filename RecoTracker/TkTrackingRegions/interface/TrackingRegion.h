#ifndef TrackingRegion_H
#define TrackingRegion_H

/** \class TrackingRegion
 * The ABC class to define the region of interest for regional seeding
 */

#include <vector>
#include <string>

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include <vector>

class DetLayer;
class HitRZCompatibility;
namespace edm { class Event; class EventSetup; }

class TrackingRegion{
public:

  typedef PixelRecoRange<float> Range;
  typedef TransientTrackingRecHit::ConstRecHitPointer Hit;
  typedef std::vector<Hit> Hits;

  /// the direction around which region is constructed 
  virtual GlobalVector direction() const = 0;

 /** The origin (centre,vertex) of the region. <BR> 
  *  The origin with bounds is ment to constraint point of the <BR>
  *  closest approach of the track to the beam line
  */
  virtual GlobalPoint  origin() const = 0;

  /// bounds the particle vertex in the transverse plane  
  virtual float originRBound() const = 0;

  /// bounds the particle vertex in the longitudinal plane 
  virtual float originZBound() const = 0;

  /// minimal pt of interest 
  virtual float ptMin()  const = 0;

   /// get hits from layer compatible with region constraints 
   virtual Hits hits(
       const edm::Event& ev, 
       const edm::EventSetup& es, 
       const ctfseeding::SeedingLayer* layer) const = 0; 

   /// utility to check eta/theta hit compatibility with region constraints 
   /// and outer hit constraint  */
  virtual HitRZCompatibility * checkRZ(const DetLayer* layer,  
				       const Hit & outerHit,
				       const edm::EventSetup& iSetup) const = 0;

  /// new region with updated vertex position 
  virtual TrackingRegion* restrictedRegion( const GlobalPoint &  originPos, 
      const float & originRBound, const float & originZBound) const = 0;

  /// clone region
      virtual TrackingRegion* clone() const = 0;
  
  virtual std::string name() const { return "TrackingRegion"; }
  virtual std::string print() const = 0;
};

#endif
