#include "SimG4CMS/Calo/interface/HGCMouseBite.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

//#define EDM_ML_DEBUG

HGCMouseBite::HGCMouseBite(const HGCalDDDConstants& hgc, const std::vector<double>& angle, double maxL, bool rot)
    : hgcons_(hgc), cut_(maxL), rot_(rot) {
  modeUV_ =
      ((hgcons_.geomMode() == HGCalGeometryMode::Hexagon8) || (hgcons_.geomMode() == HGCalGeometryMode::Hexagon8Full));
  for (auto ang : angle) {
    projXY_.push_back(std::pair<double, double>(cos(ang * CLHEP::deg), sin(ang * CLHEP::deg)));
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Creating HGCMosueBite with cut at " << cut_ << " with mode " << modeUV_ << " along "
                             << angle.size() << " axes";
  for (unsigned int k = 0; k < angle.size(); ++k)
    edm::LogVerbatim("HGCSim") << "Axis[" << k << "] " << angle[k] << " with projections " << projXY_[k].first << ":"
                               << projXY_[k].second;
#endif
}

bool HGCMouseBite::exclude(G4ThreeVector& point, int zside, int waferU, int waferV) {
  bool check(false);
  int lay = hgcons_.getLayer(point.z(), false);
  std::pair<double, double> xy =
      (modeUV_ ? hgcons_.waferPosition(lay, waferU, waferV, false) : hgcons_.waferPosition(waferU, false));
  double xx = (zside > 0) ? xy.first : -xy.first;
  double dx(0), dy(0);
  if (rot_) {
    dx = std::abs(point.y() - xy.second);
    dy = std::abs(point.x() - xx);
  } else {
    dx = std::abs(point.x() - xx);
    dy = std::abs(point.y() - xy.second);
  }
  for (auto proj : projXY_) {
    double dist = dx * proj.first + dy * proj.second;
    if (dist > cut_) {
      check = true;
      break;
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCMouseBite:: Point " << point << " zside " << zside << " wafer " << waferU << ":"
                             << waferV << " position " << xy.first << ":" << xx << ":" << xy.second << " dxy " << dx
                             << ":" << dy << " check " << check;
#endif
  return check;
}
