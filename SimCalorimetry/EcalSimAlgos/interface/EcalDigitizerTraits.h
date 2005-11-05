#ifndef EcalDigitizerTraits_h
#define EcalDigitizerTraits_h
using namespace cms;
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"

class EBDigitizerTraits {
  typedef EBDigiCollection DigiCollection;
  typedef EBDataFrame Digi;
  typedef EcalCoder ElectronicsSim;
};


class EEDigitizerTraits {
  typedef EEDigiCollection DigiCollection;
  typedef EEDataFrame Digi;
  typedef EcalCoder ElectronicsSim;
};


#endif

