#ifndef SimG4CMS_HGCalTestBeam_AHCALGEOMETRY_H
#define SimG4CMS_HGCalTestBeam_AHCALGEOMETRY_H 1

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "AHCalDetId.h"

/** \class AHCalGeometry
 *  Gets position of AHCal cell
 */

class AHCalGeometry {
public:
  /** Create geometry of AHCal */
  AHCalGeometry(edm::ParameterSet const&);
  ~AHCalGeometry() {}

  /// get maximum number of layers
  int maxDepth() const { return maxDepth_; }

  /// get the local coordinate in the plane and along depth
  std::pair<double, double> getXY(const AHCalDetId& id) const;
  double getZ(const AHCalDetId& id) const;

private:
  AHCalGeometry() = delete;
  const int maxDepth_;
  const double deltaX_, deltaY_, deltaZ_, zFirst_;
};
#endif
