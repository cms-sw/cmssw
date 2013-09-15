#ifndef SimMuL1_EtaRangeHelpers_h
#define SimMuL1_EtaRangeHelpers_h

namespace etaRangeHelpers
{
  bool isME1bEtaRegion(float eta, float eta_min = 1.64, float eta_max = 2.14);
  bool isME1abEtaRegion(float eta, float eta_min = 1.64);
  bool isME1aEtaRegion(float eta, float eta_min = 2.14);
  bool isME42EtaRegion(float eta);
  bool isME42RPCEtaRegion(float eta);
}

#endif
