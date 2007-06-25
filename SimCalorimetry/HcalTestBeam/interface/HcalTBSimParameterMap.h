#ifndef HcalTestBeam_HcalTBSimParameterMap_h
#define HcalTestBeam_HcalTBSimParameterMap_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
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
  CaloSimParameters hbParameters() const {return theHBParameters;}
  CaloSimParameters heParameters() const {return theHEParameters;}
  CaloSimParameters hoParameters() const {return theHOParameters;}

private:
  CaloSimParameters theHBParameters;
  CaloSimParameters theHEParameters;
  CaloSimParameters theHOParameters;
};


#endif

