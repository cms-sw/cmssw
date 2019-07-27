#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/HGCalTestBeam/interface/AHCalGeometry.h"

AHCalGeometry::AHCalGeometry(edm::ParameterSet const& iC)
    : maxDepth_(iC.getUntrackedParameter<int>("maxDepth", 12)),
      deltaX_(iC.getUntrackedParameter<double>("deltaX", 30.0)),
      deltaY_(iC.getUntrackedParameter<double>("deltaY", 30.0)),
      deltaZ_(iC.getUntrackedParameter<double>("deltaZ", 81.0)),
      zFirst_(iC.getUntrackedParameter<double>("zFirst", 17.6)) {
  edm::LogVerbatim("HGCSim") << "AHCalGeometry: maxDepth = " << maxDepth_ << " deltaX = " << deltaX_
                             << " deltaY = " << deltaY_ << " deltaZ = " << deltaZ_ << " zFirst = " << zFirst_;
}

std::pair<double, double> AHCalGeometry::getXY(const AHCalDetId& id) const {
  int row = id.irow();
  int col = id.icol();
  double shiftx = (col > 0) ? -0.5 * deltaX_ : 0.5 * deltaX_;
  double shifty = (row > 0) ? -0.5 * deltaY_ : 0.5 * deltaY_;
  return std::pair<double, double>(col * deltaX_ + shiftx, row * deltaY_ + shifty);
}

double AHCalGeometry::getZ(const AHCalDetId& id) const {
  int lay = id.depth();
  return (zFirst_ + (lay - 1) * deltaZ_);
}
