#ifndef TrackingRegion_H
#define TrackingRegion_H

/** \class TrackingRegion
 * The ABC class to define the region of interest for regional seeding
 */
#include <utility>
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include <string>

//#include "CommonDet/BasicDet/interface/RecHit.h"
//#include "CommonDet/DetLayout/interface/DetLayer.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
//#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
//#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "FWCore/Framework/interface/EventSetup.h"
//#include "CARF/Reco/interface/RecObj.h"

//class TrackingRegion : public RecObj {
class TrackingRegion{
public:

  typedef PixelRecoRange<float> Range;

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

  /// (signed) inverse pt range 
 /*  virtual Range  invPtRange() const = 0; */

/*   /// get hits from layer compatible with region constraints */
/*   virtual vector<RecHit> hits(const DetLayer* layer) const = 0; */






/*   /// utility to check eta/theta hit compatibility with region constraints */
/*   /// and outer hit constraint  */
/*  virtual HitRZCompatibility * checkRZ( */
/*       const DetLayer* layer, SiPixelRecHit  outerHit) const = 0; */
  virtual HitRZCompatibility * checkRZ(const DetLayer* layer,  
				       const TrackingRecHit*  outerHit,
				       const edm::EventSetup& iSetup) const = 0;
  /// new region with updated vertex position 
  virtual TrackingRegion* restrictedRegion( const GlobalPoint &  originPos, 
      const float & originRBound, const float & originZBound) const = 0;

  /// clone region
      virtual TrackingRegion* clone() const = 0;

  ///from RecObj
  static std::string const & name() 
    { static std::string local("TrackingRegion"); return local; }
  virtual std::string const & getName() const {return name();}

  
};

#endif
