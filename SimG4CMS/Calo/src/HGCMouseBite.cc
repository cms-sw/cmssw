#include "SimG4CMS/Calo/interface/HGCMouseBite.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

//#define EDM_ML_DEBUG

HGCMouseBite::HGCMouseBite(const HGCalDDDConstants& hgc,
			   const std::vector<double>& angle, double maxL, 
			   bool rot) : hgcons_(hgc), cut_(maxL), rot_(rot) {

  for (auto ang : angle) {
    projXY_.push_back(std::pair<double,double>(cos(ang*CLHEP::deg),sin(ang*CLHEP::deg)));
  }
#ifdef EDM_ML_DEBUG
  std::cout << "Creating HGCMosueBite with cut at " << cut_ << " along " 
	    << angle.size() << " axes" << std::endl;
  for (unsigned int k=0; k<angle.size(); ++k) 
    std::cout << "Axis[" << k << "] " << angle[k] << " with projections "
	      << projXY_[k].first << ":" << projXY_[k].second << std::endl;
#endif
}

bool HGCMouseBite::exclude(G4ThreeVector& point, int zside, int wafer) {
  bool check(false);
  std::pair<double,double> xy = hgcons_.waferPosition(wafer,false);
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
    double dist = dx*proj.first + dy*proj.second;
    if (dist > cut_) {check = true; break;}
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HGCMouseBite:: Point " << point << " zside " << zside
	    << " wafer " << wafer << " position " << xy.first << ":" << xx
	    << ":" << xy.second << " dxy " << dx << ":" << dy << " check "
	    << check << std::endl;
#endif
  return check;
}
