#ifndef OuterHitPhiPrediction_H
#define OuterHitPhiPrediction_H

/** predicts phi range at a given radius r */

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

#include "FWCore/Utilities/interface/Visibility.h"

#include <cassert>

class dso_internal OuterHitPhiPrediction {
public:

  using Range= PixelRecoRange<float>;

  OuterHitPhiPrediction( 
      const Range & phiAtVertex, 
      const Range & curvature, 
      float originRBound) 
    : thePhiAtVertex(phiAtVertex), theCurvature(curvature),
      theOriginRBound (originRBound) {
       assert(theCurvature.max()>0);
       assert(theCurvature.max() == -theCurvature.min()); 
      } 

  void  setTolerance(float tolerance) { theTolerance = tolerance; }

  Range operator()(float radius) const { return sym(radius);}

private:

  Range sym(float radius) const;
  Range	asym(float radius) const;


  Range thePhiAtVertex;
  Range theCurvature;
  float theOriginRBound;
  float theTolerance = 0.f;
};

#endif
