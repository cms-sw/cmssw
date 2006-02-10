#ifndef EcalSimAlgos_EcalSimParameterMap_h
#define EcalSimAlgos_EcalSimParameterMap_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"


class EcalSimParameterMap : public CaloVSimParameterMap
{
public:
  EcalSimParameterMap();
  virtual ~EcalSimParameterMap() {}

  virtual const CaloSimParameters & simParameters(const DetId & id) const;

private:
  CaloSimParameters theBarrelParameters;
  CaloSimParameters theEndcapParameters;
  CaloSimParameters theESParameters;
};

#endif

