#ifndef SimG4CMS_HGCalTestBeam_AHCALGEOMETRY_H
#define SimG4CMS_HGCalTestBeam_AHCALGEOMETRY_H 1

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4CMS/HGCalTestBeam/interface/AHCalDetId.h"
#include "Geometry/HGCalCommonData/interface/AHCalParameters.h"

/** \class AHCalGeometry
 *  Gets position of AHCal cell
 */

class AHCalGeometry {
public:
  /** Create geometry of AHCal */
  AHCalGeometry(edm::ParameterSet const&);
  AHCalGeometry() = delete;
  ~AHCalGeometry() {}

  /// get maximum number of layers
  int maxDepth() const { return ahcal_->maxDepth(); }

  /// get the local coordinate in the plane and along depth
  std::pair<double, double> getXY(const AHCalDetId& id) const;
  double getZ(const AHCalDetId& id) const;

private:
  std::unique_ptr<AHCalParameters> ahcal_;
};
#endif
