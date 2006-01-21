#ifndef EcalSimAlgos_EcalDigitizerTraits_h
#define EcalSimAlgos_EcalDigitizerTraits_h

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"

class EBDigitizerTraits {
  typedef EBDigiCollection DigiCollection;
  typedef EBDataFrame Digi;
  typedef EcalElectronicsSim ElectronicsSim;
};


class EEDigitizerTraits {
  typedef EEDigiCollection DigiCollection;
  typedef EEDataFrame Digi;
  typedef EcalElectronicsSim ElectronicsSim;
};


#endif

