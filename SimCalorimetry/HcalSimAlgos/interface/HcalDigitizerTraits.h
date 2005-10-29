#ifndef HcalDigitizerTraits_h
#define HcalDigitizerTraits_h
using namespace cms;
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"

class HBHEDigitizerTraits {
  typedef HBHEDigiCollection DigiCollection;
  typedef HBHEDataFrame Digi;
  typedef HcalElectronicsSim ElectronicsSim;
};


class HODigitizerTraits {
  typedef HODigiCollection DigiCollection;
  typedef HODataFrame Digi;
  typedef HcalElectronicsSim ElectronicsSim;
};


class HFDigitizerTraits {
  typedef HFDigiCollection DigiCollection;
  typedef HFDataFrame Digi;
  typedef HcalElectronicsSim ElectronicsSim;
};

#endif

