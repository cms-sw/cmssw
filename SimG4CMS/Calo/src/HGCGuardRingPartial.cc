#include "SimG4CMS/Calo/interface/HGCGuardRingPartial.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include <iostream>

//#define EDM_ML_DEBUG

HGCGuardRingPartial::HGCGuardRingPartial(const HGCalDDDConstants& hgc)
    : hgcons_(hgc),
      modeUV_(hgcons_.geomMode()),
      waferSize_(hgcons_.waferSize(false)),
      guardRingOffset_(hgcons_.getParameter()->guardRingOffset_) {
  offset_ = guardRingOffset_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Creating HGCGuardRingPartial with wafer size " << waferSize_ << ", Offsets "
                             << ":" << guardRingOffset_ << ":" << offset_ << ", and mode " << modeUV_;
#endif
}

bool HGCGuardRingPartial::exclude(G4ThreeVector& point, int zside, int frontBack, int layer, int waferU, int waferV) {
  bool check(false);
  if (modeUV_ == HGCalGeometryMode::Hexagon8Cassette) {
    int index = HGCalWaferIndex::waferIndex(layer, waferU, waferV);
    int partial = HGCalWaferType::getPartial(index, hgcons_.getParameter()->waferInfoMap_);
    int type = HGCalWaferType::getType(index, hgcons_.getParameter()->waferInfoMap_);
    if (partial == HGCalTypes::WaferFull) {
      return (check);
    } else {
      int orient = HGCalWaferType::getOrient(index, hgcons_.getParameter()->waferInfoMap_);
      int placement = HGCalCell::cellPlacementIndex(zside, frontBack, orient);
      double delX = 0.5 * waferSize_;
      double delY = 2 * delX / sqrt3_;
      double dx = (zside > 0) ? -point.x() : point.x();
      double dy = point.y();
      double tresh = std::abs(offset_ / cos_1[placement]);
      if (type > 0) {
        check |= std::abs(dy - (dx * tan_1[placement])) < tresh;
        check |= std::abs(dy - (dx * tan_1[placement]) + ((HGCalTypes::c10 * delY * 0.5) / cos_1[placement])) < tresh;
        check |= std::abs(dy * cot_1[placement] - (dx)) < tresh;
      } else {
        check |= std::abs((dy * cot_1[placement]) - dx + ((HGCalTypes::c22 * delX) / cos_1[placement])) < tresh;
        check |= std::abs(dy - (dx * tan_1[placement]) - ((HGCalTypes::c27 * delY) / cos_1[placement])) < tresh;
        check |= std::abs(dy - (dx * tan_1[placement]) + ((HGCalTypes::c27 * delY) / cos_1[placement])) < tresh;
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
