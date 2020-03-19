#ifndef TrackAssociator_HcalDetIdAssociator_h
#define TrackAssociator_HcalDetIdAssociator_h 1
// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      HcalDetIdAssociator
//
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
//
//

#include "CaloDetIdAssociator.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
class HcalDetIdAssociator : public CaloDetIdAssociator {
public:
  HcalDetIdAssociator() : CaloDetIdAssociator(72, 70, 0.087, nullptr){};

  HcalDetIdAssociator(int hcalReg, int nPhi, int nEta, double etaBinSize, CaloGeometry const* geom)
      : CaloDetIdAssociator(nPhi, nEta, etaBinSize, geom), hcalReg_{hcalReg} {}

  const char* name() const override { return "HCAL"; }

protected:
  int hcalReg_;
  const unsigned int getNumberOfSubdetectors() const override { return hcalReg_; }
  void getValidDetIds(unsigned int subDetectorIndex, std::vector<DetId>& validIds) const override {
    if (subDetectorIndex == 0)
      validIds = geometry_->getValidDetIds(DetId::Hcal, HcalBarrel);  //HB
    else
      validIds = geometry_->getValidDetIds(DetId::Hcal, HcalEndcap);  //HE
  };
};
#endif
