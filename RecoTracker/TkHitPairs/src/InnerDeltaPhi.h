#ifndef InnerDeltaPhi_H
#define InnerDeltaPhi_H

/** predict phi bending in layer for the tracks constratind by outer hit r-z */ 
#include <fstream>
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/GeometryVector/interface/Phi.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

class DetLayer;
class MultipleScatteringParametrisation;
template<class T> class PixelRecoRange;

class TVector3;

class InnerDeltaPhi {
public:

  InnerDeltaPhi( const DetLayer& layer,
                 const TrackingRegion & region,
                 const edm::EventSetup& iSetup,
                 bool precise = true);

  InnerDeltaPhi( const DetLayer& layer, 
		 float ptMin,  float rOrigin,
		 float zMinOrigin, float zMaxOrigin,const edm::EventSetup& iSetup,
		 bool precise = true);

   ~InnerDeltaPhi();

  float operator()( float rHit, float zHit, float errRPhi = 0.) const;
  PixelRecoRange<float> operator()( float rHit, float phiHit, float zHit, float errRPhi);
  float dphi(float rHit, float phiHit, float zHit, float errRPhi) const;

private:

  float theROrigin;
  float theRLayer;
  float theRCurvature;
  float theHitError;
  float theA;
  float theB;
  bool  theRDefined;
  float theVtxX, theVtxY, theVtxZ;
  float theVtxR;
  float thePtMin;
  MultipleScatteringParametrisation * sigma;
  bool thePrecise;

  ///AK
  Geom::Phi<float> phiIP;
  ////

  void initBarrelLayer( const DetLayer& layer);
  void initForwardLayer( const DetLayer& layer, float zMinOrigin, float zMaxOrigin);

  float innerRadius( float hitX, float hitY, float hitZ) const;
  float minRadius( float hitR, float hitZ) const {
    if (theRDefined) return theRLayer;
    else {
      float invRmin = (hitZ-theB)/theA/(hitR-theVtxR);
      return ( invRmin> 0) ? std::max( (1./invRmin+theVtxR), (double)theRLayer) : theRLayer;
    }
  }

  ////////////////////////////////
  TVector3 findTrackCenter(float rHit, float yHit, int sign)const;

  double findPhi(double x0, double y0, double r0, 
		 double x1, double y1, double r1, double phiHit)const;
  ////////////////////////////////


};

#endif
