#ifndef CastorSim_CastorSimParameterMap_h
#define CastorSim_CastorSimParameterMap_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class CastorSimParameterMap : public CaloVSimParameterMap
{
public:
  /// hardcoded default parameters
  CastorSimParameterMap();
  /// configurable parameters
  CastorSimParameterMap(const edm::ParameterSet & p);

  virtual ~CastorSimParameterMap() {}

  virtual const CaloSimParameters & simParameters(const DetId & id) const;

  /// accessors
  //CaloSimParameters castorParameters() const  {return theCastorParameters;}
    CastorSimParameters castorParameters() const 
	{
	    return theCastorParameters;
	}
    
    

  void setDbService(const CastorDbService * service);

private:
  CastorSimParameters theCastorParameters;
};

#endif

