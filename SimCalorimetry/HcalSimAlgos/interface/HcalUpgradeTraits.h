#ifndef HcalSimAlgos_HcalUpgradeTraits_h
#define HcalSimAlgos_HcalUpgradeTraits_h

#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"

class HcalUpgradeDigitizerTraits {

public:
  typedef HcalUpgradeDigiCollection DigiCollection;
  typedef HcalUpgradeDataFrame Digi;
  typedef HcalElectronicsSim ElectronicsSim;
};

#endif
