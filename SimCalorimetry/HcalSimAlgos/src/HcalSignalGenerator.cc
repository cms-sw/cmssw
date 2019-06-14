#include "SimCalorimetry/HcalSimAlgos/interface/HcalSignalGenerator.h"

//specializations

template <>
bool HcalSignalGenerator<HcalQIE10DigitizerTraits>::validDigi(
    const HcalSignalGenerator<HcalQIE10DigitizerTraits>::DIGI& digi) {
  int DigiSum = 0;
  for (int id = 0; id < digi.samples(); id++) {
    if (digi[id].adc() > 0)
      ++DigiSum;
  }
  return (DigiSum > 0);
}

template <>
CaloSamples HcalSignalGenerator<HcalQIE10DigitizerTraits>::samplesInPE(
    const HcalSignalGenerator<HcalQIE10DigitizerTraits>::DIGI& digi) {
  HcalDetId cell = digi.id();
  CaloSamples result = CaloSamples(cell, digi.samples());

  // first, check if there was an overflow in this fake digi:
  bool overflow = false;
  // find and list them

  for (int isample = 0; isample < digi.samples(); ++isample) {
    //only check error bit for non-zero ADCs
    if (digi[isample].adc() > 0 && !digi[isample].ok())
      overflow = true;
  }

  if (overflow) {  // do full conversion, go back and overwrite fake entries
    const HcalQIECoder* channelCoder = theConditions->getHcalCoder(cell);
    const HcalQIEShape* channelShape = theConditions->getHcalShape(cell);
    HcalCoderDb coder(*channelCoder, *channelShape);
    coder.adc2fC(digi, result);

    // overwrite with coded information
    for (int isample = 0; isample < digi.samples(); ++isample) {
      if (digi[isample].ok())
        result[isample] = float(digi[isample].adc()) / HcalQIE10DigitizerTraits::PreMixFactor;
    }
  } else {  // saves creating the coder, etc., every time
    // use coded information
    for (int isample = 0; isample < digi.samples(); ++isample) {
      result[isample] = float(digi[isample].adc()) / HcalQIE10DigitizerTraits::PreMixFactor;
      if (digi[isample].soi())
        result.setPresamples(isample);
    }
  }

  // translation done in fC, convert to pe:
  fC2pe(result);

  return result;
}

template <>
void HcalSignalGenerator<HcalQIE10DigitizerTraits>::fillDigis(
    const HcalSignalGenerator<HcalQIE10DigitizerTraits>::COLLECTION* digis) {
  // loop over digis, adding these to the existing maps
  for (typename COLLECTION::const_iterator it = digis->begin(); it != digis->end(); ++it) {
    QIE10DataFrame df(*it);
    // for the first signal, set the starting cap id
    if ((it == digis->begin()) && theElectronicsSim) {
      int startingCapId = df[0].capid();
      theElectronicsSim->setStartingCapId(startingCapId);
    }
    if (validDigi(df)) {
      theNoiseSignals.push_back(samplesInPE(df));
    }
  }
}

template <>
bool HcalSignalGenerator<HcalQIE11DigitizerTraits>::validDigi(
    const HcalSignalGenerator<HcalQIE11DigitizerTraits>::DIGI& digi) {
  int DigiSum = 0;
  for (int id = 0; id < digi.samples(); id++) {
    if (digi[id].adc() > 0)
      ++DigiSum;
  }
  return (DigiSum > 0);
}

template <>
CaloSamples HcalSignalGenerator<HcalQIE11DigitizerTraits>::samplesInPE(
    const HcalSignalGenerator<HcalQIE11DigitizerTraits>::DIGI& digi) {
  HcalDetId cell = digi.id();
  CaloSamples result = CaloSamples(cell, digi.samples());

  // first, check if there was an overflow in this fake digi:
  bool overflow = false;
  // find and list them

  for (int isample = 0; isample < digi.samples(); ++isample) {
    if (digi[isample].tdc() == 1) {
      overflow = true;
      break;
    }
  }

  if (overflow) {  // do full conversion, go back and overwrite fake entries
    const HcalQIECoder* channelCoder = theConditions->getHcalCoder(cell);
    const HcalQIEShape* channelShape = theConditions->getHcalShape(cell);
    HcalCoderDb coder(*channelCoder, *channelShape);
    coder.adc2fC(digi, result);

    // overwrite with coded information
    for (int isample = 0; isample < digi.samples(); ++isample) {
      if (digi[isample].tdc() == 0)
        result[isample] = float(digi[isample].adc()) / HcalQIE11DigitizerTraits::PreMixFactor;
    }
  } else {  // saves creating the coder, etc., every time
    // use coded information
    for (int isample = 0; isample < digi.samples(); ++isample) {
      result[isample] = float(digi[isample].adc()) / HcalQIE11DigitizerTraits::PreMixFactor;
      if (digi[isample].soi())
        result.setPresamples(isample);
    }
  }

  // translation done in fC, convert to pe:
  fC2pe(result);

  return result;
}

template <>
void HcalSignalGenerator<HcalQIE11DigitizerTraits>::fillDigis(
    const HcalSignalGenerator<HcalQIE11DigitizerTraits>::COLLECTION* digis) {
  // loop over digis, adding these to the existing maps
  for (typename COLLECTION::const_iterator it = digis->begin(); it != digis->end(); ++it) {
    QIE11DataFrame df(*it);
    // for the first signal, set the starting cap id
    if ((it == digis->begin()) && theElectronicsSim) {
      int startingCapId = df[0].capid();
      theElectronicsSim->setStartingCapId(startingCapId);
    }
    if (validDigi(df)) {
      theNoiseSignals.push_back(samplesInPE(df));
    }
  }
}
