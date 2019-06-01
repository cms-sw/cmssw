// to make hits in EB/EE/HC
#include "SimG4CMS/Calo/interface/HcalNumberingFromPS.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <iostream>
#include <iomanip>

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

HcalNumberingFromPS::HcalNumberingFromPS(const edm::ParameterSet& conf) {
  etaTable_ = conf.getParameter<std::vector<double> >("EtaTable");
  phibin_ = conf.getParameter<std::vector<double> >("PhiBin");
  phioff_ = conf.getParameter<std::vector<double> >("PhiOffset");
  etaMin_ = conf.getParameter<std::vector<int> >("EtaMin");
  etaMax_ = conf.getParameter<std::vector<int> >("EtaMax");
  etaHBHE_ = conf.getParameter<int>("EtaHBHE");
  depthHBHE_ = conf.getParameter<std::vector<int> >("DepthHBHE");
  depth29Mx_ = conf.getParameter<int>("Depth29Max");
  rMinHO_ = conf.getParameter<double>("RMinHO");
  zHO_ = conf.getParameter<std::vector<double> >("ZHO");
  const double deg = M_PI / 180.0;
  for (auto& phi : phibin_)
    phi *= deg;
  for (auto& phi : phioff_)
    phi *= deg;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HcalNumberingFromPS:: EtaTable with " << etaTable_.size() << " elements";
  for (unsigned k = 0; k < etaTable_.size(); ++k)
    edm::LogVerbatim("HcalSim") << "EtaTable[" << k << "] = " << etaTable_[k];
  edm::LogVerbatim("HcalSim") << "HcalNumberingFromPS:: PhiBin with " << phibin_.size() << " elements";
  for (unsigned k = 0; k < phibin_.size(); ++k)
    edm::LogVerbatim("HcalSim") << "PhiBin[" << k << "] = " << phibin_[k];
  edm::LogVerbatim("HcalSim") << "HcalNumberingFromPS:: PhiOff with " << phioff_.size() << " elements";
  for (unsigned k = 0; k < phioff_.size(); ++k)
    edm::LogVerbatim("HcalSim") << "PhiOff[" << k << "] = " << phioff_[k];
  edm::LogVerbatim("HcalSim") << "HcalNumberingFromPS:: EtaMin/EtaMax with " << etaMin_.size() << ":" << etaMax_.size()
                              << " elements";
  for (unsigned k = 0; k < etaMin_.size(); ++k)
    edm::LogVerbatim("HcalSim") << "EtaMin[" << k << "] = " << etaMin_[k] << " EtaMax[" << k << "] = " << etaMax_[k];
  edm::LogVerbatim("HcalSim") << "HcalNumberingFromPS:: EtaHBHE " << etaHBHE_ << " DepthHBHE " << depthHBHE_[0] << ":"
                              << depthHBHE_[1] << " RMinHO " << rMinHO_ << " zHO with " << zHO_.size() << " elements";
  for (unsigned k = 0; k < zHO_.size(); ++k)
    edm::LogVerbatim("HcalSim") << "ZHO[" << k << "] = " << zHO_[k];
#endif

  segmentation_.resize(nEtas_);
  for (int ring = 0; ring < nEtas_; ++ring) {
    char name[10];
    snprintf(name, 10, "Eta%d", ring + 1);
    if (ring > 0) {
      segmentation_[ring] = conf.getUntrackedParameter<std::vector<int> >(name, segmentation_[ring - 1]);
    } else {
      segmentation_[ring] = conf.getUntrackedParameter<std::vector<int> >(name);
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "HcalNumberingFromPS:: Ring " << ring + 1 << " with " << segmentation_[ring].size()
                                << " layers";
    for (unsigned int k = 0; k < segmentation_[ring].size(); ++k)
      edm::LogVerbatim("HcalSim") << "Layer[" << k << "] = " << segmentation_[ring][k];
#endif
  }
}

HcalNumberingFromDDD::HcalID HcalNumberingFromPS::unitID(int det,
                                                         int layer,
                                                         int depth,
                                                         const math::XYZVectorD& pos) const {
  int subdet = ((det == 3) ? static_cast<int>(HcalBarrel) : static_cast<int>(HcalEndcap));
  std::pair<int, int> deteta = getEta(subdet, pos);
  std::pair<int, int> iphi = getPhi(deteta.first, deteta.second, pos.Phi());
  int newDepth(depth);
  int zside = ((pos.z() > 0) ? 1 : 0);
  if (deteta.first == static_cast<int>(HcalBarrel)) {
    newDepth = segmentation_[deteta.second - 1][layer - 1];
    if ((deteta.second == etaHBHE_) && (newDepth > depthHBHE_[0]))
      newDepth = depthHBHE_[0];
  } else if (deteta.first == static_cast<int>(HcalEndcap)) {
    newDepth = segmentation_[deteta.second - 1][layer - 1];
    if ((deteta.second == etaHBHE_) && (newDepth < depthHBHE_[1]))
      newDepth = depthHBHE_[1];
    if ((deteta.second == etaMax_[1]) && (newDepth > depth29Mx_))
      --(deteta.second);
  } else if (deteta.first == static_cast<int>(HcalOuter)) {
    newDepth = 4;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HcalNumberingFromPS:: det " << det << ":" << subdet << ":" << deteta.first
                              << "\t Eta " << pos.Eta() << ":" << deteta.second << "\t Phi " << pos.Phi() << ":"
                              << iphi.first << ":" << iphi.second << "\t Layer|Depth " << layer << ":" << depth << ":"
                              << newDepth;
#endif
  return HcalNumberingFromDDD::HcalID(deteta.first, zside, newDepth, deteta.second, iphi.first, iphi.second, layer);
}

std::pair<int, int> HcalNumberingFromPS::getEta(const int& det, const math::XYZVectorD& pos) const {
  int ieta(1);
  int subdet(det);
  double eta = std::abs(pos.Eta());
  if (pos.Rho() > rMinHO_) {
    subdet = static_cast<int>(HcalOuter);
    double z = std::abs(pos.z());
    if (z > zHO_[3]) {
      if (eta <= etaTable_[10])
        eta = etaTable_[10] + 0.001;
    } else if (z > zHO_[1]) {
      if (eta <= etaTable_[4])
        eta = etaTable_[4] + 0.001;
    }
  }
  for (unsigned int i = 1; i < etaTable_.size(); i++) {
    if (eta < etaTable_[i]) {
      ieta = i;
      break;
    }
  }

  if ((subdet == static_cast<int>(HcalBarrel)) || (subdet == static_cast<int>(HcalOuter))) {
    if (ieta > etaMax_[0])
      ieta = etaMax_[0];
  } else if (det == static_cast<int>(HcalEndcap)) {
    if (ieta <= etaMin_[1])
      ieta = etaMin_[1];
  }
  return std::make_pair(subdet, ieta);
}

std::pair<int, int> HcalNumberingFromPS::getPhi(const int& det, const int& ieta, const double& phi) const {
  double fioff = ((det == static_cast<int>(HcalEndcap)) ? phioff_[1] : phioff_[0]);
  double fibin = phibin_[ieta - 1];
  int nphi = int((2._pi + 0.1 * fibin) / fibin);
  double hphi = phi + fioff;
  if (hphi < 0)
    hphi += (2._pi);
  int iphi = int(hphi / fibin) + 1;
  if (iphi > nphi)
    iphi = 1;
  const double fiveDegInRad = 5._deg;
  int units = int(fibin / fiveDegInRad + 0.5);
  if (units < 1)
    units = 1;
  int iphi_skip = iphi;
  if (units == 2)
    iphi_skip = (iphi - 1) * 2 + 1;
  else if (units == 4)
    iphi_skip = (iphi - 1) * 4 - 1;
  if (iphi_skip < 0)
    iphi_skip += 72;
  return std::make_pair(iphi, iphi_skip);
}
