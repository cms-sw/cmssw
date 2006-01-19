#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "CLHEP/Random/RandFlat.h"


namespace cms {

  HcalElectronicsSim::HcalElectronicsSim(HcalAmplifier * amplifier, const HcalCoderFactory * coderFactory)
    : theAmplifier(amplifier),
      theCoderFactory(coderFactory)
  {
  }


  template<class Digi> 
  void HcalElectronicsSim::convert(CaloSamples & frame, Digi & result, bool addNoise) {
    result.setSize(frame.size());
    if(addNoise) theAmplifier->amplify(frame);
    theCoderFactory->coder(frame.id())->fC2adc(frame, result, theStartingCapId);
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

  
  void HcalElectronicsSim::newEvent() {
    // pick a new starting Capacitor ID
    theStartingCapId = RandFlat::shootInt(4);
    theAmplifier->setStartingCapId(theStartingCapId);
  }

}

