#ifndef TrackingRegionBase_H
#define TrackingRegionBase_H

/** \class TrackingRegionBase
 * kinematic data common to 
 * some concreate implementations of TrackingRegion.
 */
#include <utility>

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <sstream>
class BarrelDetLayer;
class ForwardDetLayer; 

class TrackingRegionBase : public TrackingRegion {

public:

  TrackingRegionBase( const GlobalVector & direction,
                  const GlobalPoint &  originPos,
                  const Range        & invPtRange,
                  const float &        originRBound,
                  const float &        originZBound)
    : theDirection( direction), theVertexPos( originPos), 
      theInvPtRange( invPtRange), theVertexRBound( originRBound),
      theVertexZBound( originZBound) { }    

  /** dummy ctor */
  TrackingRegionBase() { }

  /** dtor */
  virtual ~TrackingRegionBase() { }

  /// the direction around which region is constructed 
  virtual GlobalVector direction() const { return theDirection; } 

 /** The origin (centre,vertex) of the region. <BR> 
  *  The origin with bounds is ment to constraint point of the <BR>
  *  closest approach of the track to the beam line
  */
  virtual GlobalPoint  origin() const { return theVertexPos; }

  /// bounds the particle vertex in the transverse plane  
  virtual float originRBound() const { return theVertexRBound; }

  /// bounds the particle vertex in the longitudinal plane 
  virtual float originZBound() const { return theVertexZBound; }

  /// minimal pt of interest 
  virtual float ptMin()  const { 
    return 1./std::max(fabs(theInvPtRange.max()), fabs(theInvPtRange.min())); 
  } 

  /// inverse pt range 
  virtual Range        invPtRange() const { return theInvPtRange; }

 

  /// utility to check eta/theta hit compatibility with region constraints
  /// and outer hit constraint
    virtual HitRZCompatibility * checkRZ(const DetLayer* layer,  
					 const Hit & outerHit,
					 const edm::EventSetup& iSetup) const = 0;

  /// clone region with new vertex position
  virtual TrackingRegionBase* restrictedRegion( const GlobalPoint &  originPos,
      const float & originRBound, const float & originZBound) const {
      TrackingRegionBase* restr = clone();
      restr->theVertexPos = originPos;
      restr->theVertexRBound = originRBound;
      restr->theVertexZBound = originZBound;
      return restr;
  } 

  virtual TrackingRegionBase* clone() const = 0;

  virtual std::string print() const {
    std::ostringstream str;
    str << name() <<" dir:"<<theDirection<<" vtx:"<<theVertexPos 
        <<" dr:"<<theVertexRBound<<" dz:"<<theVertexZBound<<" pt:"<<1./theInvPtRange.max();
    return str.str();
  }

  virtual void setDirection(const GlobalVector & dir ) { theDirection = dir; }

private:
  
  GlobalVector theDirection;
  GlobalPoint  theVertexPos;
  Range        theInvPtRange;
  float        theVertexRBound;
  float        theVertexZBound;

};

#endif
