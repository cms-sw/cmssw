#ifndef HcalSimParameterMap_h
#define HcalSimParameterMap_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"

namespace cms {
  class HcalSimParameterMap : public CaloVSimParameterMap
  {
  public:
    HcalSimParameterMap();
    virtual ~HcalSimParameterMap() {}

    virtual const CaloSimParameters & simParameters(const DetId & id) const;

    /// accessors
    CaloSimParameters hbheParameters() const {return theHBHEParameters;}
    CaloSimParameters hoParameters() const  {return theHOParameters;}
    CaloSimParameters hfParameters1() const  {return theHFParameters1;}
    CaloSimParameters hfParameters2() const  {return theHFParameters2;}

  private:
    CaloSimParameters theHBHEParameters;
    CaloSimParameters theHOParameters;
    CaloSimParameters theHFParameters1;
    CaloSimParameters theHFParameters2;
  };
}

#endif

