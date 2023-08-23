#ifndef SimG4CMS_HGCGuardRing_h
#define SimG4CMS_HGCGuardRing_h

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "G4ThreeVector.hh"

#include <vector>

class HGCGuardRing {
public:
  HGCGuardRing(const HGCalDDDConstants& hgc);
  bool exclude(G4ThreeVector& point, int zside, int frontBack, int layer, int waferU, int waferV);
  static bool insidePolygon(double x, double y, const std::vector<std::pair<double, double> >& xyv);

private:
  static constexpr double sqrt3_ = 1.732050807568877;  // std::sqrt(3.0) in double precision
  const HGCalDDDConstants& hgcons_;
  const HGCalGeometryMode::GeometryMode modeUV_;
  const double waferSize_, sensorSizeOffset_, guardRingOffset_;
  double offset_, xmax_, ymax_;
};

#endif  // HGCGuardRing_h
