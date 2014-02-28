#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"

#include "CLHEP/Random/RandFlat.h"


HcalElectronicsSim::HcalElectronicsSim(HcalAmplifier * amplifier, const HcalCoderFactory * coderFactory)
  : theAmplifier(amplifier),
    theCoderFactory(coderFactory),
    theStartingCapId(0),
    theStartingCapIdIsRandom(true)
{
}


HcalElectronicsSim::~HcalElectronicsSim() {
}


void HcalElectronicsSim::setDbService(const HcalDbService * service) {
  //  theAmplifier->setDbService(service);
  theTDC.setDbService(service);
}

template<class Digi> 
void HcalElectronicsSim::convert(CaloSamples & frame, Digi & result, CLHEP::HepRandomEngine* engine) {
  result.setSize(frame.size());
  theAmplifier->amplify(frame, engine);
  theCoderFactory->coder(frame.id())->fC2adc(frame, result, theStartingCapId);
}


void HcalElectronicsSim::analogToDigital(CLHEP::HepRandomEngine* engine, CaloSamples & lf, HBHEDataFrame & result) {
  convert<HBHEDataFrame>(lf, result, engine);
}


void HcalElectronicsSim::analogToDigital(CLHEP::HepRandomEngine* engine, CaloSamples & lf, HODataFrame & result) {
  convert<HODataFrame>(lf, result, engine);
}


void HcalElectronicsSim::analogToDigital(CLHEP::HepRandomEngine* engine, CaloSamples & lf, HFDataFrame & result) {
  convert<HFDataFrame>(lf, result, engine);
}

void HcalElectronicsSim::analogToDigital(CLHEP::HepRandomEngine* engine, CaloSamples & lf, ZDCDataFrame & result) {
  convert<ZDCDataFrame>(lf, result, engine);
}


void HcalElectronicsSim::analogToDigital(CLHEP::HepRandomEngine* engine, CaloSamples & lf,
					 HcalUpgradeDataFrame & result) {
  convert<HcalUpgradeDataFrame>(lf, result, engine);
//   std::cout << HcalDetId(lf.id()) << ' ' << lf;
  theTDC.timing(lf, result, engine);
}

void HcalElectronicsSim::newEvent(CLHEP::HepRandomEngine* engine) {
  // pick a new starting Capacitor ID
  if(theStartingCapIdIsRandom)
  {
    theStartingCapId = CLHEP::RandFlat::shootInt(engine, 4);
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

