#include <memory>



#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/HGCalTestBeam/interface/AHCalGeometry.h"

AHCalGeometry::AHCalGeometry(edm::ParameterSet const& iC) { ahcal_ = std::make_unique<AHCalParameters>(iC); }

std::pair<double, double> AHCalGeometry::getXY(const AHCalDetId& id) const {
  int row = id.irow();
  int col = id.icol();
  double shiftx = (col > 0) ? -0.5 * ahcal_->deltaX() : 0.5 * ahcal_->deltaX();
  double shifty = (row > 0) ? -0.5 * ahcal_->deltaY() : 0.5 * ahcal_->deltaY();
  return std::pair<double, double>(col * ahcal_->deltaX() + shiftx, row * ahcal_->deltaY() + shifty);
}

double AHCalGeometry::getZ(const AHCalDetId& id) const {
  int lay = id.depth();
  return (ahcal_->zFirst() + (lay - 1) * ahcal_->deltaZ());
}
