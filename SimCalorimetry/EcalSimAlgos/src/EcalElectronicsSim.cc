#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"

namespace cms { 

  void EcalElectronicsSim::run(const CaloSamples & lf, EBDataFrame & result) {
    convert<EBDataFrame>(lf, result);
  }


  void EcalElectronicsSim::run(const CaloSamples & lf, EEDataFrame & result) {
    convert<EEDataFrame>(lf, result);
  }

}


