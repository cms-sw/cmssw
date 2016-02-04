#ifndef HcalTestBeam_HcalTBSimParameterMap_h
#define HcalTestBeam_HcalTBSimParameterMap_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalTBSimParameterMap : public CaloVSimParameterMap {

public:
  /// hardcoded default parameters
  HcalTBSimParameterMap();
  /// configurable parameters
  HcalTBSimParameterMap(const edm::ParameterSet & p);

  virtual ~HcalTBSimParameterMap() {}

  virtual const CaloSimParameters & simParameters(const DetId & id) const;

  /// accessors
  HcalSimParameters hbParameters() const {return theHBParameters;}
  HcalSimParameters heParameters() const {return theHEParameters;}
  HcalSimParameters hoParameters() const {return theHOParameters;}

private:
  HcalSimParameters theHBParameters;
  HcalSimParameters theHEParameters;
  HcalSimParameters theHOParameters;
};


#endif

