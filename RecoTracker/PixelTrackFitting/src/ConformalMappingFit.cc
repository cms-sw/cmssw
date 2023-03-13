#include "ConformalMappingFit.h"

using namespace std;

template <class T>
T sqr(T t) {
  return t * t;
}

ConformalMappingFit::ConformalMappingFit(const std::vector<PointXY>& hits,
                                         const std::vector<float>& errRPhi2,
                                         const Rotation* rot)
    : theRotation(rot), myRotation(rot == nullptr) {
  typedef ConformalMappingFit::MappedPoint<double> PointUV;
  int hits_size = hits.size();
  for (int i = 0; i < hits_size; i++) {
    if (!theRotation)
      findRot(hits[i]);
    PointUV point(hits[i], 1. / errRPhi2[i], theRotation);
    theFit.addPoint(point.u(), point.v(), point.weight());
  }
}

void ConformalMappingFit::findRot(const PointXY& p) {
  myRotation = true;
  typedef Rotation::GlobalVector GlobalVector;  // ::GlobalVector is float!
  GlobalVector aX = GlobalVector(p.x(), p.y(), 0.).unit();
  GlobalVector aY(-aX.y(), aX.x(), 0.);
  GlobalVector aZ(0., 0., 1.);
  theRotation = new Rotation(aX, aY, aZ);
}

ConformalMappingFit::~ConformalMappingFit() {
  if (myRotation)
    delete theRotation;
}

double ConformalMappingFit::phiRot() const { return atan2(theRotation->xy(), theRotation->xx()); }

Measurement1D ConformalMappingFit::curvature() const {
  double val = fabs(2. * theFit.parA());
  double err = 2. * sqrt(theFit.varAA());
  return Measurement1D(val, err);
}

Measurement1D ConformalMappingFit::directionPhi() const {
  double val = phiRot() + atan(theFit.parB());
  double err = sqrt(theFit.varBB());
  return Measurement1D(val, err);
}

Measurement1D ConformalMappingFit::impactParameter() const {
  double val = -theFit.parC();
  double err = sqrt(theFit.varCC());
  return Measurement1D(val, err);
}

int ConformalMappingFit::charge() const { return (theFit.parA() > 0.) ? -1 : 1; }
