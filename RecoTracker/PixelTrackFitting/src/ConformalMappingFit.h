#ifndef RecoTracker_PixelTrackFitting_src_ConformalMappingFit_h
#define RecoTracker_PixelTrackFitting_src_ConformalMappingFit_h

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

#include "ParabolaFit.h"
#include <vector>

namespace edm {
  class ParameterSet;
}

class ConformalMappingFit {
public:
  typedef TkRotation<double> Rotation;
  typedef Basic2DVector<double> PointXY;

  ConformalMappingFit(const std::vector<PointXY> &hits,
                      const std::vector<float> &errRPhi2,
                      const Rotation *rotation = nullptr);

  ~ConformalMappingFit();

  Measurement1D curvature() const;
  Measurement1D directionPhi() const;
  Measurement1D impactParameter() const;

  int charge() const;
  double chi2() const { return theFit.chi2(); }

  const Rotation *rotation() const { return theRotation; }

  void fixImpactParmaeter(double ip) { theFit.fixParC(ip); }
  void skipErrorCalculation() { theFit.skipErrorCalculationByDefault(); }

private:
  double phiRot() const;
  void findRot(const PointXY &);

private:
  const Rotation *theRotation;
  bool myRotation;
  ParabolaFit theFit;

  template <class T>
  class MappedPoint {
  public:
    typedef Basic2DVector<T> PointXY;
    MappedPoint() : theU(0), theV(0), theW(0), pRot(0) {}
    MappedPoint(const T &aU, const T &aV, const T &aWeight, const TkRotation<T> *aRot)
        : theU(aU), theV(aV), theW(aWeight), pRot(aRot) {}
    MappedPoint(const PointXY &point, const T &weight, const TkRotation<T> *aRot) : pRot(aRot) {
      T radius2 = point.mag2();
      Basic3DVector<T> rotated = (*pRot) * point;
      theU = rotated.x() / radius2;
      theV = rotated.y() / radius2;
      theW = weight * radius2 * radius2;
    }
    T u() const { return theU; }
    T v() const { return theV; }
    T weight() const { return theW; }
    PointXY unmap() const {
      T invRadius2 = theU * theU + theV * theV;
      Basic3DVector<T> tmp = (*pRot).multiplyInverse(Basic2DVector<T>(theU, theV));
      return PointXY(tmp.x() / invRadius2, tmp.y() / invRadius2);
    }
    T unmappedWeight() const {
      T invRadius2 = theU * theU + theV * theV;
      return theW * invRadius2 * invRadius2;
    }

  private:
    T theU, theV, theW;
    const TkRotation<T> *pRot;
  };
};

#endif
