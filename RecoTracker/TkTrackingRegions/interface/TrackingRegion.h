#ifndef TrackingRegion_H
#define TrackingRegion_H

/** \class TrackingRegion
 * kinematic data common to 
 * some concreate implementations of TrackingRegion.
 */

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/HitEtaCheck.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRCheck.h"
#include "RecoTracker/TkTrackingRegions/interface/HitZCheck.h"


#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <utility>

#include <sstream>

#include <vector>
#include <string>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class DetLayer;
class HitRZCompatibility;
class BarrelDetLayer;
class ForwardDetLayer;

namespace edm { class Event;  }

class TrackingRegion {
public:

public:
  virtual ~TrackingRegion(){}
  typedef PixelRecoRange<float> Range;
  typedef TransientTrackingRecHit::ConstRecHitPointer Hit;
  typedef std::vector<Hit> Hits;


public:

  TrackingRegion( const GlobalVector & direction,
                  const GlobalPoint &  originPos,
                  const Range        & invPtRange,
                  const float &        originRBound,
                  const float &        originZBound)
    : theDirection( direction), theVertexPos( originPos), 
      theInvPtRange( invPtRange),
      thePtMin(1.f/std::max( std::abs(invPtRange.max()), std::abs(invPtRange.min()) )),
      theVertexRBound( originRBound),
      theVertexZBound( originZBound) { }    


  /// the direction around which region is constructed 
  GlobalVector const & direction() const { return theDirection; } 

 /** The origin (centre,vertex) of the region. <BR> 
  *  The origin with bounds is ment to constraint point of the <BR>
  *  closest approach of the track to the beam line
  */
  GlobalPoint  const & origin() const { return theVertexPos; }

  /// bounds the particle vertex in the transverse plane  
  float originRBound() const { return theVertexRBound; }

  /// bounds the particle vertex in the longitudinal plane 
  float originZBound() const { return theVertexZBound; }

  /// minimal pt of interest 
  float ptMin()  const { return thePtMin;}

  /// inverse pt range 
  Range invPtRange() const { return theInvPtRange; }


  /// utility to check eta/theta hit compatibility with region constraints
  /// and outer hit constraint
  virtual HitRZCompatibility * checkRZ(const DetLayer* layer,  
				       const Hit &  outerHit,
				       const edm::EventSetup& iSetup,
				       const DetLayer* outerlayer=0, 
				       float lr=0, float gz=0, float dr=0, float dz=0) const = 0;


/// get hits from layer compatible with region constraints 
    virtual Hits hits(
        const edm::Event& ev, 
        const edm::EventSetup& es, 
        const ctfseeding::SeedingLayer* layer) const = 0; 

  /// clone region with new vertex position
  TrackingRegion* restrictedRegion( const GlobalPoint &  originPos,
      const float & originRBound, const float & originZBound) const {
      TrackingRegion* restr = clone();
      restr->theVertexPos = originPos;
      restr->theVertexRBound = originRBound;
      restr->theVertexZBound = originZBound;
      return restr;
  } 

  virtual TrackingRegion* clone() const = 0;

  virtual std::string name() const { return "TrackingRegion"; }
  virtual std::string print() const {
    std::ostringstream str;
    str << name() <<" dir:"<<theDirection<<" vtx:"<<theVertexPos 
        <<" dr:"<<theVertexRBound<<" dz:"<<theVertexZBound<<" pt:"<<1./theInvPtRange.max();
    return str.str();
  }

  void setDirection(const GlobalVector & dir ) { theDirection = dir; }

private:
  
  GlobalVector theDirection;
  GlobalPoint  theVertexPos;
  Range        theInvPtRange;
  float        thePtMin;
  float        theVertexRBound;
  float        theVertexZBound;

};

using TrackingRegionBase = TrackingRegion;

#endif
