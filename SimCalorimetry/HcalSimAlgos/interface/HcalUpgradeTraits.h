#ifndef HcalSimAlgos_HcalUpgradeTraits_h
#define HcalSimAlgos_HcalUpgradeTraits_h

#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"

class HcalUpgradeDigitizerTraits {

public:
  typedef HcalUpgradeDigiCollection DigiCollection;
  typedef HcalUpgradeDataFrame Digi;
  typedef HcalElectronicsSim ElectronicsSim;
  static constexpr double PreMixFactor = 10.0;
  static const unsigned PreMixBits = 126;
};

#endif
