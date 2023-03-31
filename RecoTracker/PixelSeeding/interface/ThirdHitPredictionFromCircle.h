#ifndef ThirdHitPredictionFromCircle_H
#define ThirdHitPredictionFromCircle_H

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

class ThirdHitPredictionFromCircle {
public:
  using Scalar = double;
  typedef PixelRecoRange<float> Range;
  typedef Basic2DVector<Scalar> Vector2D;
  typedef Basic2DVector<float> VectorF2D;

  ThirdHitPredictionFromCircle(const GlobalPoint &P1, const GlobalPoint &P2, float tolerance);

  float phi(float curvature, float radius) const;
  float angle(float curvature, float radius) const;

  Range operator()(Range curvature, float radius) const;

  Range curvature(double transverseIP) const;
  double curvature(const Vector2D &thirdPoint) const;
  double transverseIP(const Vector2D &thirdPoint) const;

  // like PixelRecoLineRZ, but makes use of the bending computation
  // from the circle fit to get an actual Helix propagation
  class HelixRZ {
  public:
    using Vector2D = ThirdHitPredictionFromCircle::Vector2D;
    using Scalar = ThirdHitPredictionFromCircle::Scalar;
    // using Vector2D=ThirdHitPredictionFromCircle::VectorF2D;
    // using Scalar=float;

    HelixRZ() : circle(nullptr) {}
    HelixRZ(const ThirdHitPredictionFromCircle *icircle, double iz1, double z2, double curv);

    Scalar zAtR(Scalar r) const;
    Scalar rAtZ(Scalar z) const;

    static double maxCurvature(const ThirdHitPredictionFromCircle *circle, double z1, double z2, double z3);

  private:
    const ThirdHitPredictionFromCircle *circle;
    Vector2D center;
    Scalar curvature, radius, z1, seg, dzdu;
  };

private:
  friend class HelixRZ;

  Scalar invCenterOnAxis(const Vector2D &thirdPoint) const;

  Vector2D p1, center, axis;
  Scalar delta, delta2;
  float theTolerance;
};

#endif
