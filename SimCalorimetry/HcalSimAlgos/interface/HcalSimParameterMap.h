#ifndef HcalSimAlgos_HcalSimParameterMap_h
#define HcalSimAlgos_HcalSimParameterMap_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFSimParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include <unordered_set>

class HcalSimParameterMap : public CaloVSimParameterMap
{
public:
  /// hardcoded default parameters
  HcalSimParameterMap();
  /// configurable parameters
  HcalSimParameterMap(const edm::ParameterSet & p);

  ~HcalSimParameterMap() override {}

  const CaloSimParameters & simParameters(const DetId & id) const override;

  /// accessors
  const HcalSimParameters & hbParameters() const {return theHBParameters;}
  const HcalSimParameters & heParameters() const {return theHEParameters;}
  const HcalSimParameters & hoParameters() const  {return theHOParameters;}
  const HFSimParameters & hfParameters1() const  {return theHFParameters1;}
  const HFSimParameters & hfParameters2() const  {return theHFParameters2;}
  const HFSimParameters & zdcParameters() const  {return theZDCParameters;}

  void setDbService(const HcalDbService * service);

  void setFrameSize(const DetId & detId, int frameSize);
  
  void setHOZecotekDetIds(const std::unordered_set<DetId> & ids)
  {
    theHOZecotekDetIds = ids;
  }
  void setHOHamamatsuDetIds(const std::unordered_set<DetId> & ids)
  {
    theHOHamamatsuDetIds = ids;
  }

private:
  void setFrameSize(CaloSimParameters & parameters, int frameSize);

  HcalSimParameters theHBParameters;
  HcalSimParameters theHEParameters;
  HcalSimParameters theHOParameters;
  HcalSimParameters theHOZecotekSiPMParameters;
  HcalSimParameters theHOHamamatsuSiPMParameters;
  HFSimParameters theHFParameters1;
  HFSimParameters theHFParameters2;
  HFSimParameters theZDCParameters;
  std::unordered_set<DetId> theHOZecotekDetIds;
  std::unordered_set<DetId> theHOHamamatsuDetIds;
};

#endif

