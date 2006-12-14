#ifndef HcalSimAlgos_HcalSimParameterMap_h
#define HcalSimAlgos_HcalSimParameterMap_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class HcalSimParameterMap : public CaloVSimParameterMap
{
public:
  /// hardcoded default parameters
  HcalSimParameterMap();
  /// configurable parameters
  HcalSimParameterMap(const edm::ParameterSet & p);

  virtual ~HcalSimParameterMap() {}

  virtual const CaloSimParameters & simParameters(const DetId & id) const;

  /// accessors
  HcalSimParameters hbParameters() const {return theHBParameters;}
  HcalSimParameters heParameters() const {return theHEParameters;}
  HcalSimParameters hoParameters() const  {return theHOParameters;}
  CaloSimParameters hfParameters1() const  {return theHFParameters1;}
  CaloSimParameters hfParameters2() const  {return theHFParameters2;}

  void setDbService(const HcalDbService * service);

private:
  HcalSimParameters theHBParameters;
  HcalSimParameters theHEParameters;
  HcalSimParameters theHOParameters;
  CaloSimParameters theHFParameters1;
  CaloSimParameters theHFParameters2;
};


#endif

