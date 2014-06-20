#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"
#include "CLHEP/Random/RandFlat.h"
#include <math.h>

HcalElectronicsSim::HcalElectronicsSim(HcalAmplifier * amplifier, const HcalCoderFactory * coderFactory, bool PreMixing)
  : theAmplifier(amplifier),
    theCoderFactory(coderFactory),
    theRandFlat(0),
    theStartingCapId(0),
    theStartingCapIdIsRandom(false),
    PreMixDigis(PreMixing)
{
}


HcalElectronicsSim::~HcalElectronicsSim() {
  if (theRandFlat) delete theRandFlat;
}


void HcalElectronicsSim::setRandomEngine(CLHEP::HepRandomEngine & engine) {
  theRandFlat = new CLHEP::RandFlat(engine);
  theAmplifier->setRandomEngine(engine);
  theTDC.setRandomEngine(engine);
}


void HcalElectronicsSim::setDbService(const HcalDbService * service) {
  //  theAmplifier->setDbService(service);
  theTDC.setDbService(service);
}

template<class Digi> 
void HcalElectronicsSim::convert(CaloSamples & frame, Digi & result) {
  result.setSize(frame.size());
  theAmplifier->amplify(frame);

    theCoderFactory->coder(frame.id())->fC2adc(frame, result, theStartingCapId);
}


void HcalElectronicsSim::analogToDigital(CaloSamples & lf, HBHEDataFrame & result) {
  convert<HBHEDataFrame>(lf, result);  
  if(PreMixDigis) {
    for(int isample = 0; isample !=lf.size(); ++isample) {
      uint16_t theADC = round(10.0*lf[isample]);
      unsigned capId = result[isample].capid();

      if(theADC > 126) {
	uint16_t keepADC = result[isample].adc();
	HcalQIESample mysamp(keepADC, capId, 0, 0, true, true); // set error bit as a flag
	result.setSample(isample, HcalQIESample(keepADC, capId, 0, 0, true, true) );
      }
      else {
	result.setSample(isample, HcalQIESample(theADC, capId, 0, 0)  );  // preserve fC, no noise
	HcalQIESample mysamp(theADC, capId, 0, 0);
      }
    }
  }
}


void HcalElectronicsSim::analogToDigital(CaloSamples & lf, HODataFrame & result) {
  convert<HODataFrame>(lf, result);
  if(PreMixDigis) {
    for(int isample = 0; isample !=lf.size(); ++isample) {
      uint16_t theADC = round(10.0*lf[isample]);
      unsigned capId = result[isample].capid();

      if(theADC > 126) {
	uint16_t keepADC = result[isample].adc();
	HcalQIESample mysamp(keepADC, capId, 0, 0, true, true);// set error bit as a flag
	result.setSample(isample, HcalQIESample(keepADC, capId, 0, 0, true, true) );
      }
      else {
	result.setSample(isample, HcalQIESample(theADC, capId, 0, 0)  );  // preserve fC, no noise
	HcalQIESample mysamp(theADC, capId, 0, 0);
      }
    }
  }
}


void HcalElectronicsSim::analogToDigital(CaloSamples & lf, HFDataFrame & result) {
  convert<HFDataFrame>(lf, result);  //not enough dynamic range with simple multiplication
  if(PreMixDigis) {
    for(int isample = 0; isample !=lf.size(); ++isample) {
      uint16_t theADC = round(10.0*lf[isample]);
      unsigned capId = result[isample].capid();

      if(theADC > 126) {
	uint16_t keepADC = result[isample].adc();
	HcalQIESample mysamp(keepADC, capId, 0, 0, true, true);// set error bit as a flag
	result.setSample(isample, HcalQIESample(keepADC, capId, 0, 0, true, true) );
      }
      else {
	result.setSample(isample, HcalQIESample(theADC, capId, 0, 0)  );  // preserve fC, no noise
	HcalQIESample mysamp(theADC, capId, 0, 0);
      }
    }
  }
}

void HcalElectronicsSim::analogToDigital(CaloSamples & lf, ZDCDataFrame & result) {
  convert<ZDCDataFrame>(lf, result);  
  if(PreMixDigis) {
    for(int isample = 0; isample !=lf.size(); ++isample) {
      uint16_t theADC = round(10.0*lf[isample]);
      unsigned capId = result[isample].capid();

      if(theADC > 126) {
	uint16_t keepADC = result[isample].adc();
	HcalQIESample mysamp(keepADC, capId, 0, 0, true, true);//set error bit as a flag
	result.setSample(isample, HcalQIESample(keepADC, capId, 0, 0, true, true) );
      }
      else {
	result.setSample(isample, HcalQIESample(theADC, capId, 0, 0)  );  // preserve fC, no noise
	HcalQIESample mysamp(theADC, capId, 0, 0);
      }
    }
  }
}


void HcalElectronicsSim::analogToDigital(CaloSamples & lf, 
					 HcalUpgradeDataFrame & result) {
  convert<HcalUpgradeDataFrame>(lf, result);
//   std::cout << HcalDetId(lf.id()) << ' ' << lf;
  theTDC.timing(lf, result);
}

void HcalElectronicsSim::newEvent() {
  // pick a new starting Capacitor ID
  if(theStartingCapIdIsRandom)
  {
    theStartingCapId = theRandFlat->fireInt(4);
    theAmplifier->setStartingCapId(theStartingCapId);
  }
}


void HcalElectronicsSim::setStartingCapId(int startingCapId)
{
  theStartingCapId = startingCapId;
  theAmplifier->setStartingCapId(theStartingCapId);
  // turns off random capIDs forever for this instance
  theStartingCapIdIsRandom = false;
}

