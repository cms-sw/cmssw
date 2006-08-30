#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "CLHEP/Random/RandFlat.h"



HcalElectronicsSim::HcalElectronicsSim(HcalAmplifier * amplifier, const HcalCoderFactory * coderFactory)
  : theAmplifier(amplifier),
    theCoderFactory(coderFactory),
    theStartingCapId(0)
{
}


template<class Digi> 
void HcalElectronicsSim::convert(CaloSamples & frame, Digi & result) {
  result.setSize(frame.size());
  theAmplifier->amplify(frame);
  theCoderFactory->coder(frame.id())->fC2adc(frame, result, theStartingCapId);
}


void HcalElectronicsSim::analogToDigital(CaloSamples & lf, HBHEDataFrame & result) {
  convert<HBHEDataFrame>(lf, result);
}


void HcalElectronicsSim::analogToDigital(CaloSamples & lf, HODataFrame & result) {
  convert<HODataFrame>(lf, result);
}


void HcalElectronicsSim::analogToDigital(CaloSamples & lf, HFDataFrame & result) {
  convert<HFDataFrame>(lf, result);
}


void HcalElectronicsSim::newEvent() {
  // pick a new starting Capacitor ID
  theStartingCapId = RandFlat::shootInt(4);
  theAmplifier->setStartingCapId(theStartingCapId);
}

