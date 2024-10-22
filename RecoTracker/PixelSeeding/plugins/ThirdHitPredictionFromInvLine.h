#ifndef RecoTracker_PixelSeeding_plugins_ThirdHitPredictionFromInvLine_h
#define RecoTracker_PixelSeeding_plugins_ThirdHitPredictionFromInvLine_h
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

#include <cassert>

class ThirdHitPredictionFromInvLine {
public:
  typedef TkRotation<double> Rotation;
  typedef PixelRecoRange<float> Range;

  ThirdHitPredictionFromInvLine(const GlobalPoint& P1,
                                const GlobalPoint& P2,
                                double errorRPhiP1 = 1.,
                                double errorRPhiP2 = 1.);

  int size() const { return nPoints; }

  void add(const GlobalPoint& p, double erroriRPhi = 1.);

  void remove(const GlobalPoint& p, double erroriRPhi = 1.);

  GlobalPoint crossing(double radius) const;

  double curvature() const {
    assert(hasParameters);
    return theCurvatureValue;
  }

  double errorCurvature() const {
    assert(hasParameters);
    return theCurvatureError;
  }

  double chi2() const {
    assert(hasParameters);
    return theChi2;
  }

  void print() const;

private:
  template <class T>
  class MappedPoint {
  public:
    MappedPoint() : theU(0), theV(0), pRot(0) {}
    MappedPoint(const T& aU, const T& aV, const TkRotation<T>* aRot) : theU(aU), theV(aV), pRot(aRot) {}
    MappedPoint(const Basic2DVector<T>& point, const TkRotation<T>* aRot) : pRot(aRot) {
      T radius2 = point.mag2();
      Basic3DVector<T> rotated = (*pRot) * point;
      theU = rotated.x() / radius2;
      theV = rotated.y() / radius2;
    }
    T u() const { return theU; }
    T v() const { return theV; }
    Basic2DVector<T> unmap() const {
      T invRadius2 = theU * theU + theV * theV;
      Basic3DVector<T> tmp = (*pRot).multiplyInverse(Basic2DVector<T>(theU, theV));
      return Basic2DVector<T>(tmp.x() / invRadius2, tmp.y() / invRadius2);
    }

  private:
    T theU, theV;
    const TkRotation<T>* pRot;
  };

private:
  typedef MappedPoint<double> PointUV;
  void add(const ThirdHitPredictionFromInvLine::PointUV& point, double weight);
  void check();

private:
  Rotation theRotation;
  int nPoints;
  long double theSum, theSumU, theSumUU, theSumV, theSumUV, theSumVV;
  bool hasParameters;
  double theCurvatureValue, theCurvatureError, theChi2;
};

#endif
