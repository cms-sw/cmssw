#ifndef SimG4CMS_HGCGuardRingPartial_h
#define SimG4CMS_HGCGuardRingPartial_h

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "G4ThreeVector.hh"

#include <vector>
#include <array>

class HGCGuardRingPartial {
public:
  HGCGuardRingPartial(const HGCalDDDConstants& hgc);
  bool exclude(G4ThreeVector& point, int zside, int frontBack, int layer, int waferU, int waferV);
  static bool insidePolygon(double x, double y, const std::vector<std::pair<double, double> >& xyv);

private:
  static constexpr double sqrt3_ = 1.732050807568877;  // std::sqrt(3.0) in double precision
  const HGCalDDDConstants& hgcons_;
  const HGCalGeometryMode::GeometryMode modeUV_;
  const double waferSize_, guardRingOffset_;
  static constexpr std::array<double, 12> tan_1 = {{-sqrt3_, sqrt3_, 0.0, -sqrt3_,  sqrt3_, 0.0, sqrt3_, -sqrt3_, 0.0, sqrt3_, -sqrt3_, 0.0}};
  static constexpr std::array<double, 12> cos_1 = {{0.5, -0.5, -1.0, -0.5, 0.5, 1.0, -0.5, 0.5, 1.0, 0.5, -0.5, -1.0}};
  static constexpr std::array<double, 12> cot_1 = {{sqrt3_, -sqrt3_, 0.0, sqrt3_, -sqrt3_, 0.0, -sqrt3_, sqrt3_, 0.0, -sqrt3_, sqrt3_, 0.0}};
  double offset_, xmax_, ymax_;
};

#endif  // HGCGuardRing_h
