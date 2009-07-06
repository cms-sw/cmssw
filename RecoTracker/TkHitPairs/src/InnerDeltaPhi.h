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
                 bool precise = true,
                 float extraTolerance = 0.);

   ~InnerDeltaPhi();

  float operator()( float rHit, float zHit, float errRPhi) const;
   
  PixelRecoRange<float> operator()( float rHit, float phiHit, float zHit, float errRPhi) const;

private:

  float theROrigin;
  float theRLayer;
  float theThickness;

  float theRCurvature;
  float theExtraTolerance;
  float theA;
  float theB;
  bool  theRDefined;

  Point2D theVtx;
  float theVtxZ;
  float thePtMin;
  MultipleScatteringParametrisation * sigma;
  bool thePrecise;

private:

  void initBarrelLayer( const DetLayer& layer);
  void initForwardLayer( const DetLayer& layer, float zMinOrigin, float zMaxOrigin);

  PixelRecoRange<float> phiRange( const Point2D & hitXY, float zHit, float errRPhi) const;
  float minRadius( float hitR, float hitZ) const;

};

#endif
