#ifndef InnerDeltaPhi_H
#define InnerDeltaPhi_H

/** predict phi bending in layer for the tracks constratind by outer hit r-z */ 
#include <fstream>
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class DetLayer;
template<class T> class PixelRecoRange;

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"

class dso_hidden InnerDeltaPhi {
public:

  typedef Basic2DVector<float> Point2D;

  InnerDeltaPhi( const DetLayer& outlayer,const DetLayer& layer,
                 const TrackingRegion & region,
                 const edm::EventSetup& iSetup,
                 bool precise = true,
                 float extraTolerance = 0.f);


  PixelRecoRange<float> operator()( float xHit, float yHit, float zHit, float errRPhi) const {
    return phiRange( Point2D(xHit,yHit), zHit, errRPhi); 
  }

private:

  bool theRDefined;
  bool thePrecise;
  int ol;

  float theROrigin;
  float theRLayer;
  float theThickness;

  float theRCurvature;
  float theExtraTolerance;
  float theA;
  float theB;

  float theVtxZ;
  float thePtMin;

  Point2D theVtx;


  MultipleScatteringParametrisation sigma;


private:

  void initBarrelLayer( const DetLayer& layer);
  void initForwardLayer( const DetLayer& layer, float zMinOrigin, float zMaxOrigin);

  PixelRecoRange<float> phiRange( const Point2D & hitXY, float zHit, float errRPhi) const;
  float minRadius( float hitR, float hitZ) const;

};

#endif
