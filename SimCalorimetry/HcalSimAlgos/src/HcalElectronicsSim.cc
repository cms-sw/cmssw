#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalNoisifier.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"

namespace cms {

  HcalElectronicsSim::HcalElectronicsSim(CaloVNoisifier * noisifier, HcalCoder * coder)
    : theNoisifier(noisifier),
      theCoder(coder)
  {
  }

  void HcalElectronicsSim::analogToDigital(CaloSamples & lf, HBHEDataFrame & result, bool addNoise) {
    convert<HBHEDataFrame>(lf, result, addNoise);
  }


  void HcalElectronicsSim::analogToDigital(CaloSamples & lf, HODataFrame & result, bool addNoise) {
    convert<HODataFrame>(lf, result, addNoise);
  }


  void HcalElectronicsSim::analogToDigital(CaloSamples & lf, HFDataFrame & result, bool addNoise) {
    convert<HFDataFrame>(lf, result, addNoise);
  }

}

