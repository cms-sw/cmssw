#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "SimCalorimetry/CastorSim/src/CastorAmplifier.h"
#include "SimCalorimetry/CastorSim/src/CastorCoderFactory.h"
#include "SimCalorimetry/CastorSim/src/CastorElectronicsSim.h"

#include "CLHEP/Random/RandFlat.h"

CastorElectronicsSim::CastorElectronicsSim(CastorAmplifier *amplifier, const CastorCoderFactory *coderFactory)
    : theAmplifier(amplifier), theCoderFactory(coderFactory), theStartingCapId(0) {}

CastorElectronicsSim::~CastorElectronicsSim() {}

template <class Digi>
void CastorElectronicsSim::convert(CaloSamples &frame, Digi &result, CLHEP::HepRandomEngine *engine) {
  result.setSize(frame.size());
  theAmplifier->amplify(frame, engine);
  theCoderFactory->coder(frame.id())->fC2adc(frame, result, theStartingCapId);
}

void CastorElectronicsSim::analogToDigital(CLHEP::HepRandomEngine *engine, CaloSamples &lf, CastorDataFrame &result) {
  convert<CastorDataFrame>(lf, result, engine);
}

void CastorElectronicsSim::newEvent(CLHEP::HepRandomEngine *engine) {
  // pick a new starting Capacitor ID
  theStartingCapId = CLHEP::RandFlat::shootInt(engine, 4);
  theAmplifier->setStartingCapId(theStartingCapId);
}
