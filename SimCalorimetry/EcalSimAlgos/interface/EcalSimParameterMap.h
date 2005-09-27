#ifndef EcalSimParameterMap_h
#define EcalSimParameterMap_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"

namespace cms {

  class EcalSimParameterMap : public CaloVSimParameterMap
  {
  public:
    EcalSimParameterMap();
    virtual ~EcalSimParameterMap() {}

    virtual const CaloSimParameters & simParameters(const DetId & id) const;

  private:
    CaloSimParameters theBarrelParameters;
    CaloSimParameters theEndcapParameters;
};

}

#endif

