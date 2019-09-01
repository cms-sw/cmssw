#ifndef CastorSim_CastorDigitizerTraits_h
#define CastorSim_CastorDigitizerTraits_h

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "SimCalorimetry/CastorSim/src/CastorElectronicsSim.h"

class CastorDigitizerTraits {
public:
  typedef CastorDigiCollection DigiCollection;
  typedef CastorDataFrame Digi;
  typedef CastorElectronicsSim ElectronicsSim;
};

#endif
