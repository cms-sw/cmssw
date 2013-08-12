#ifndef SimMuL1_Helpers_h
#define SimMuL1_Helpers_h

namespace helpers
{
bool
isME1bEtaRegion(float eta, float eta_min = 1.64, float eta_max = 2.14)
{
  return fabs(eta) >= eta_min && fabs(eta) <= eta_max;
}

bool
isME1abEtaRegion(float eta, float eta_min = 1.64)
{
  return fabs(eta) >= eta_min;
}

bool 
isME1aEtaRegion(float eta, float eta_min = 2.14)
{
  return fabs(eta) >= eta_min;
}

bool 
isME42EtaRegion(float eta)
{
  return fabs(eta)>=1.2499 && fabs(eta)<=1.8;
}

bool 
isME42RPCEtaRegion(float eta)
{
  return fabs(eta)>=1.2499 && fabs(eta)<=1.6;
}

}

#endif








