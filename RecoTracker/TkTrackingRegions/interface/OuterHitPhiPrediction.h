#ifndef OuterHitPhiPrediction_H
#define OuterHitPhiPrediction_H

/** predicts phi range at a given radius r */

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"


class OuterHitPhiPrediction {
public:

  typedef PixelRecoRange<float> Range;
  typedef TkTrackingRegionsMargin<float> Margin;

  OuterHitPhiPrediction( 
      const Range & phiAtVertex, 
      const Range & curvature, 
      float originRBound, 
      const Margin & tolerance = Margin(0,0))
    : thePhiAtVertex(phiAtVertex), theCurvature(curvature),
      theOriginRBound (originRBound), theTolerance(tolerance) { } 

  void  setTolerance(const Margin & tolerance) { theTolerance = tolerance; }
  Range operator()(float radius) const;

private:
  Range thePhiAtVertex;
  Range theCurvature;
  float theOriginRBound;
  Margin theTolerance;
};

#endif
