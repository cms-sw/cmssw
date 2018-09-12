#ifndef SimPPS_PPSPixelDigiProducer_RPix_Charge_Share_H
#define SimPPS_PPSPixelDigiProducer_RPix_Charge_Share_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include "Geometry/VeryForwardGeometry/interface/CTPPSPixelSimTopology.h"
#include "SimPPS/PPSPixelDigiProducer/interface/RPixSignalPoint.h"

class RPixChargeShare
{
public:
  RPixChargeShare(const edm::ParameterSet &params, uint32_t det_id);
  std::map<unsigned short, double, std::less<unsigned short> > Share(const std::vector<RPixSignalPoint> &charge_map);

private:
  uint32_t det_id_;
  std::vector<double> signalCoupling_;
  std::map<unsigned short, double, std::less<unsigned short> > thePixelChargeMap;
  CTPPSPixelSimTopology theRPixDetTopology_;
  CTPPSPixelIndices pxlInd;
  const int pxlRowSize = pxlInd.getDefaultRowDetSize();
  const int pxlColSize = pxlInd.getDefaultColDetSize();
  int no_of_pixels_;

  double sqrt_2;
  int verbosity_;

  std::string ChargeMapFile2E_[4];
  double chargeMap2E[4][60][40];
  const int xBinMax[4]={29,59,29,59}; // X upper bins of the charge maps [0:3]
  const int yBinMax[4]={19,19,39,39}; // Y upper bins of the charge maps [0:3]

};

#endif
