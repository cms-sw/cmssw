#ifndef InnerDeltaPhi_H
#define InnerDeltaPhi_H

/** predict phi bending in layer for the tracks constratind by outer hit r-z */ 
#include <fstream>
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

class DetLayer;
class MultipleScatteringParametrisation;
template<class T> class PixelRecoRange;

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"

class InnerDeltaPhi {
public:

  typedef Basic2DVector<double> Point2D;

  InnerDeltaPhi( const DetLayer& layer,
                 const TrackingRegion & region,
                 const edm::EventSetup& iSetup,
                 bool precise = true);

   ~InnerDeltaPhi();

  float operator()( float rHit, float zHit, float errRPhi) const;
   
  PixelRecoRange<float> operator()( float rHit, float phiHit, float zHit, float errRPhi) const
  {
//     float phiM =  operator()( rHit,zHit,errRPhi); 
//     return PixelRecoRange<float>(phiHit-phiM,phiHit+phiM);

       Point2D hitXY( rHit*cos(phiHit), rHit*sin(phiHit));
       return phiRange(hitXY,zHit,errRPhi);
  }

private:

  float theROrigin;
  float theRLayer;
  float theThickness;

  float theRCurvature;
  float theHitError;
  float theA;
  float theB;
  bool  theRDefined;

  Point2D theVtx;
  float theVtxZ;
  float thePtMin;
  MultipleScatteringParametrisation * sigma;
  bool thePrecise;

  void initBarrelLayer( const DetLayer& layer);
  void initForwardLayer( const DetLayer& layer, float zMinOrigin, float zMaxOrigin);

  PixelRecoRange<float> phiRange( const Point2D & hitXY, float zHit, float errRPhi) const;
  float innerRadius( float hitX, float hitY, float hitZ) const;
  float minRadius( float hitR, float hitZ) const {
    if (theRDefined) return theRLayer;
    else {
      float invRmin = (hitZ-theB)/theA/hitR;
      return ( invRmin> 0) ? std::max( 1./invRmin, (double)theRLayer) : theRLayer;
    }
  }

};

#endif
