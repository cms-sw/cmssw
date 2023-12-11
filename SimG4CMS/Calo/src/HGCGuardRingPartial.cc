#include "SimG4CMS/Calo/interface/HGCGuardRingPartial.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include <iostream>
#include <array>

//#define EDM_ML_DEBUG

HGCGuardRingPartial::HGCGuardRingPartial(const HGCalDDDConstants& hgc)
    : hgcons_(hgc),
      modeUV_(hgcons_.geomMode()),
      v17OrLess_(hgcons_.v17OrLess()),
      waferSize_(hgcons_.waferSize(false)),
      guardRingOffset_(hgcons_.getParameter()->guardRingOffset_) {
  offset_ = guardRingOffset_;
  c22_ = (v17OrLess_) ? HGCalTypes::c22O : HGCalTypes::c22;
  c27_ = (v17OrLess_) ? HGCalTypes::c27O : HGCalTypes::c27;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Creating HGCGuardRingPartial with wafer size " << waferSize_ << ", Offsets "
                             << ":" << guardRingOffset_ << ":" << offset_ << ", and mode " << modeUV_
                             << " coefficients " << c22_ << ":" << c27_;
#endif
}

bool HGCGuardRingPartial::exclude(G4ThreeVector& point, int zside, int frontBack, int layer, int waferU, int waferV) {
  bool check(false);
  if ((modeUV_ == HGCalGeometryMode::Hexagon8Cassette) || (modeUV_ == HGCalGeometryMode::Hexagon8CalibCell)) {
    int index = HGCalWaferIndex::waferIndex(layer, waferU, waferV);
    int partial = HGCalWaferType::getPartial(index, hgcons_.getParameter()->waferInfoMap_);
    int type = HGCalWaferType::getType(index, hgcons_.getParameter()->waferInfoMap_);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCGuardRingPatial:: Layer " << layer << " wafer " << waferU << ":" << waferV
                               << " index " << index << " partial " << partial << " type " << type;
#endif
    if (partial == HGCalTypes::WaferFull) {
      return (check);
    } else if (partial < 0) {
      return true;
    } else {
      int orient = HGCalWaferType::getOrient(index, hgcons_.getParameter()->waferInfoMap_);
      int placement = HGCalCell::cellPlacementIndex(zside, frontBack, orient);
      double dx = point.x();
      double dy = point.y();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "HGCGuardRingPatial:: orient " << orient << " placement " << placement << " dx "
                                 << dx << " dy " << dy;
#endif
      if (type > 0) {
        for (int ii = HGCalTypes::WaferPartLDOffset;
             ii < (HGCalTypes::WaferPartLDOffset + HGCalTypes::WaferPartLDCount);
             ii++) {
          std::array<double, 4> criterion = HGCalWaferMask::maskCut(ii, placement, waferSize_, offset_, v17OrLess_);
          check |= std::abs(criterion[0] * dy + criterion[1] * dx + criterion[2]) < criterion[3];
        }
      } else {
        for (int ii = HGCalTypes::WaferPartHDOffset;
             ii < (HGCalTypes::WaferPartHDOffset + HGCalTypes::WaferPartHDCount);
             ii++) {
          std::array<double, 4> criterion = HGCalWaferMask::maskCut(ii, placement, waferSize_, offset_, v17OrLess_);
          check |= std::abs(criterion[0] * dy + criterion[1] * dx + criterion[2]) < criterion[3];
        }
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCGuardRingPartial:: Point " << point << " zside " << zside << " layer " << layer
                               << " wafer " << waferU << ":" << waferV << " partial type " << partial << " type "
                               << type << " check " << check;
#endif
  }
  return check;
}
