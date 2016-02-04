#include "SimCalorimetry/CastorSim/src/CastorElectronicsSim.h"
#include "SimCalorimetry/CastorSim/src/CastorAmplifier.h"
#include "SimCalorimetry/CastorSim/src/CastorCoderFactory.h"
#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "CLHEP/Random/RandFlat.h"



CastorElectronicsSim::CastorElectronicsSim(CastorAmplifier * amplifier, const CastorCoderFactory * coderFactory)
  : theAmplifier(amplifier),
    theCoderFactory(coderFactory),
    theRandFlat(0),
    theStartingCapId(0)
{
}


CastorElectronicsSim::~CastorElectronicsSim()
{
  delete theRandFlat;
}


void CastorElectronicsSim::setRandomEngine(CLHEP::HepRandomEngine & engine)
{
  theRandFlat = new CLHEP::RandFlat(engine);
}


template<class Digi> 
void CastorElectronicsSim::convert(CaloSamples & frame, Digi & result) {
  result.setSize(frame.size());
  theAmplifier->amplify(frame);
  theCoderFactory->coder(frame.id())->fC2adc(frame, result, theStartingCapId);
}

void CastorElectronicsSim::analogToDigital(CaloSamples & lf, CastorDataFrame & result) {
  convert<CastorDataFrame>(lf, result);
}

void CastorElectronicsSim::newEvent() {
  // pick a new starting Capacitor ID
  theStartingCapId = theRandFlat->fireInt(4);
  theAmplifier->setStartingCapId(theStartingCapId);
}

