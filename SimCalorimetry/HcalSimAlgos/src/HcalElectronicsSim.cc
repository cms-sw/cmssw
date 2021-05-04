#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTDC.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "CLHEP/Random/RandFlat.h"
#include <cmath>

HcalElectronicsSim::HcalElectronicsSim(const HcalSimParameterMap* parameterMap,
                                       HcalAmplifier* amplifier,
                                       const HcalCoderFactory* coderFactory,
                                       bool PreMixing)
    : theParameterMap(parameterMap),
      theAmplifier(amplifier),
      theCoderFactory(coderFactory),
      theStartingCapId(0),
      theStartingCapIdIsRandom(true),
      PreMixDigis(PreMixing) {}

HcalElectronicsSim::~HcalElectronicsSim() {}

void HcalElectronicsSim::setDbService(const HcalDbService* service) {
  //  theAmplifier->setDbService(service);
  theTDC.setDbService(service);
}

template <class Digi>
void HcalElectronicsSim::convert(CaloSamples& frame, Digi& result, CLHEP::HepRandomEngine* engine) {
  result.setSize(frame.size());
  theAmplifier->amplify(frame, engine);
  theCoderFactory->coder(frame.id())->fC2adc(frame, result, theStartingCapId);
}

template <>
void HcalElectronicsSim::convert<QIE10DataFrame>(CaloSamples& frame,
                                                 QIE10DataFrame& result,
                                                 CLHEP::HepRandomEngine* engine) {
  theAmplifier->amplify(frame, engine);
  theCoderFactory->coder(frame.id())->fC2adc(frame, result, theStartingCapId);
}

template <>
void HcalElectronicsSim::convert<QIE11DataFrame>(CaloSamples& frame,
                                                 QIE11DataFrame& result,
                                                 CLHEP::HepRandomEngine* engine) {
  theAmplifier->amplify(frame, engine);
  theCoderFactory->coder(frame.id())->fC2adc(frame, result, theStartingCapId);
}

template <class Digi>
void HcalElectronicsSim::premix(CaloSamples& frame, Digi& result, double preMixFactor, unsigned preMixBits) {
  for (int isample = 0; isample != frame.size(); ++isample) {
    uint16_t theADC = round(preMixFactor * frame[isample]);
    unsigned capId = result[isample].capid();

    if (theADC > preMixBits) {
      uint16_t keepADC = result[isample].adc();
      result.setSample(isample, HcalQIESample(keepADC, capId, 0, 0, true, true));  // set error bit as a flag
    } else {
      result.setSample(isample, HcalQIESample(theADC, capId, 0, 0));  // preserve fC, no noise
    }
  }
}

template <>
void HcalElectronicsSim::premix<QIE10DataFrame>(CaloSamples& frame,
                                                QIE10DataFrame& result,
                                                double preMixFactor,
                                                unsigned preMixBits) {
  for (int isample = 0; isample != frame.size(); ++isample) {
    uint16_t theADC = round(preMixFactor * frame[isample]);
    unsigned capId = result[isample].capid();
    bool ok = true;

    if (theADC > preMixBits) {
      theADC = result[isample].adc();
      ok = false;  // set error bit as a flag
    }

    result.setSample(
        isample, theADC, result[isample].le_tdc(), result[isample].te_tdc(), capId, result[isample].soi(), ok);
  }
}

template <>
void HcalElectronicsSim::premix<QIE11DataFrame>(CaloSamples& frame,
                                                QIE11DataFrame& result,
                                                double preMixFactor,
                                                unsigned preMixBits) {
  for (int isample = 0; isample != frame.size(); ++isample) {
    uint16_t theADC = round(preMixFactor * frame[isample]);
    int tdcErrorBit = 0;

    if (theADC > preMixBits) {
      theADC = result[isample].adc();
      tdcErrorBit = 1;  //use TDC bits for error bit
    }

    result.setSample(isample, theADC, tdcErrorBit, result[isample].soi());
  }
}

template <class Digi>
void HcalElectronicsSim::analogToDigitalImpl(
    CLHEP::HepRandomEngine* engine, CaloSamples& lf, Digi& result, double preMixFactor, unsigned preMixBits) {
  convert<Digi>(lf, result, engine);
  if (PreMixDigis)
    premix(lf, result, preMixFactor, preMixBits);
}

//TODO:
//HcalTDC extension for QIE10? and QIE11?

void HcalElectronicsSim::analogToDigital(
    CLHEP::HepRandomEngine* engine, CaloSamples& lf, HBHEDataFrame& result, double preMixFactor, unsigned preMixBits) {
  analogToDigitalImpl(engine, lf, result, preMixFactor, preMixBits);
}

void HcalElectronicsSim::analogToDigital(
    CLHEP::HepRandomEngine* engine, CaloSamples& lf, HODataFrame& result, double preMixFactor, unsigned preMixBits) {
  analogToDigitalImpl(engine, lf, result, preMixFactor, preMixBits);
}

void HcalElectronicsSim::analogToDigital(
    CLHEP::HepRandomEngine* engine, CaloSamples& lf, HFDataFrame& result, double preMixFactor, unsigned preMixBits) {
  analogToDigitalImpl(engine, lf, result, preMixFactor, preMixBits);
}

void HcalElectronicsSim::analogToDigital(
    CLHEP::HepRandomEngine* engine, CaloSamples& lf, ZDCDataFrame& result, double preMixFactor, unsigned preMixBits) {
  analogToDigitalImpl(engine, lf, result, preMixFactor, preMixBits);
}

void HcalElectronicsSim::analogToDigital(
    CLHEP::HepRandomEngine* engine, CaloSamples& lf, QIE10DataFrame& result, double preMixFactor, unsigned preMixBits) {
  analogToDigitalImpl(engine, lf, result, preMixFactor, preMixBits);
}

void HcalElectronicsSim::analogToDigital(
    CLHEP::HepRandomEngine* engine, CaloSamples& lf, QIE11DataFrame& result, double preMixFactor, unsigned preMixBits) {
  analogToDigitalImpl(engine, lf, result, preMixFactor, preMixBits);
  if (!PreMixDigis) {
    const HcalSimParameters& pars = static_cast<const HcalSimParameters&>(theParameterMap->simParameters(lf.id()));
    if (pars.threshold_currentTDC() > 0.) {
      HcalTDC theTDC((pars.threshold_currentTDC()));
      theTDC.timing(lf, result);
    }
  }
}

void HcalElectronicsSim::newEvent(CLHEP::HepRandomEngine* engine) {
  // pick a new starting Capacitor ID
  if (theStartingCapIdIsRandom) {
    theStartingCapId = CLHEP::RandFlat::shootInt(engine, 4);
    theAmplifier->setStartingCapId(theStartingCapId);
  }
}

void HcalElectronicsSim::setStartingCapId(int startingCapId) {
  theStartingCapId = startingCapId;
  theAmplifier->setStartingCapId(theStartingCapId);
  // turns off random capIDs forever for this instance
  theStartingCapIdIsRandom = false;
}
