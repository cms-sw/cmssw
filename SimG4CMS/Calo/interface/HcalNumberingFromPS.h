#ifndef SimG4CMS_HcalNumberingFromPS_h
#define SimG4CMS_HcalNumberingFromPS_h

#include "Geometry/CaloGeometry/interface/CaloNumberingScheme.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#
#include <cstdint>
#include <vector>

class HcalNumberingFromPS {
public:
  HcalNumberingFromPS(const edm::ParameterSet&);
  ~HcalNumberingFromPS() {}

  HcalNumberingFromDDD::HcalID unitID(int det, int layer, int depth, const math::XYZVectorD& pos) const;
  std::pair<int, int> getEta(const int& det, const math::XYZVectorD& pos) const;
  std::pair<int, int> getPhi(const int& det, const int& ieta, const double& phi) const;

private:
  static const int nEtas_ = 29;
  std::vector<double> etaTable_, phibin_, phioff_;
  std::vector<int> etaMin_, etaMax_, depthHBHE_;
  int etaHBHE_, depth29Mx_;
  double rMinHO_;
  std::vector<double> zHO_;
  std::vector<std::vector<int> > segmentation_;
};

#endif
