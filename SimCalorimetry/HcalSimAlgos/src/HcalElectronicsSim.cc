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

  void HcalElectronicsSim::run(CaloSamples & lf, HBHEDataFrame & result) {
    convert<HBHEDataFrame>(lf, result);
  }


  void HcalElectronicsSim::run(CaloSamples & lf, HODataFrame & result) {
    convert<HODataFrame>(lf, result);
  }


  void HcalElectronicsSim::run(CaloSamples & lf, HFDataFrame & result) {
    convert<HFDataFrame>(lf, result);
  }

}

