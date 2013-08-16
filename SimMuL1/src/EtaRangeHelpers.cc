#include "GEMCode/SimMuL1/interface/EtaRangeHelpers.h"

#include <cmath>

bool
etaRangeHelpers::isME1bEtaRegion(float eta, float eta_min, float eta_max)
{
  return fabs(eta) >= eta_min && fabs(eta) <= eta_max;
}

bool
etaRangeHelpers::isME1abEtaRegion(float eta, float eta_min)
{
  return fabs(eta) >= eta_min;
}

bool 
etaRangeHelpers::isME1aEtaRegion(float eta, float eta_min)
{
  return fabs(eta) >= eta_min;
}

bool 
etaRangeHelpers::isME42EtaRegion(float eta)
{
  return fabs(eta)>=1.2499 && fabs(eta)<=1.8;
}

bool 
etaRangeHelpers::isME42RPCEtaRegion(float eta)
{
  return fabs(eta)>=1.2499 && fabs(eta)<=1.6;
}








