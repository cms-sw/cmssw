#ifndef SimMuL1_Helpers_h
#define SimMuL1_Helpers_h

namespace triggerHelpers
{
  bool isME1bEtaRegion(float eta, float eta_min = 1.64, float eta_max = 2.14);
  bool isME1abEtaRegion(float eta, float eta_min = 1.64);
  bool isME1aEtaRegion(float eta, float eta_min = 2.14);
  bool isME42EtaRegion(float eta);
  bool isME42RPCEtaRegion(float eta);
}

#endif
