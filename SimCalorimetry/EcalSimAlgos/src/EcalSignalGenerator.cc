#include "SimCalorimetry/EcalSimAlgos/interface/EcalSignalGenerator.h"

template <>
CaloSamples EcalSignalGenerator<EBDigitizerTraits>::samplesInPE(const DIGI &digi) {
  // calibration, for future reference:  (same block for all Ecal types)
  //EcalDetId cell = digi.id();
  //         const EcalCalibrations& calibrations=conditions->getEcalCalibrations(cell);
  //const EcalQIECoder* channelCoder = theConditions->getEcalCoder (cell);
  //const EcalQIEShape* channelShape = theConditions->getEcalShape (cell);
  //EcalCoderDb coder (*channelCoder, *channelShape);
  //CaloSamples result;
  //coder.adc2fC(digi, result);
  //fC2pe(result);

  DetId detId = digi.id();

  double Emax = fullScaleEnergy(detId);
  double LSB[NGAINS + 1];

  //double icalconst = findIntercalibConstant( detId );

  double icalconst = 1.;  // find the correct value.

  const EcalIntercalibConstantMCMap &icalMap = ical->getMap();
  EcalIntercalibConstantMCMap::const_iterator icalit = icalMap.find(detId);
  if (icalit != icalMap.end()) {
    icalconst = (*icalit);
  }

  double peToA = peToAConversion(detId);

  const std::vector<float> gainRatios = GetGainRatios(detId);

  for (unsigned int igain(0); igain <= NGAINS; ++igain) {
    LSB[igain] = 0.;
    if (igain > 0)
      LSB[igain] = Emax / (MAXADC * gainRatios[igain]);
  }

  //    std::cout << " intercal, LSBs, egains " << icalconst << " " << LSB[0] << " " << LSB[1] << " " << gainRatios[0] << " " << gainRatios[1] << " " << Emax << std::endl;

  CaloSamples result(detId, digi.size());

  // correction factor for premixed sample: ratio of laser corrections
  double correction_factor_for_premixed_sample_transparency = 1.0;
  double value_LC = 1.;
  if (m_timeDependent) {
    if (detId.subdetId() != EcalSubdetector::EcalPreshower) {
      auto cache = m_valueLCCache_LC.find(detId);
      if (cache != m_valueLCCache_LC.end()) {
        value_LC = cache->second;
      } else {
        value_LC = findLaserConstant_LC(detId);
        m_valueLCCache_LC.emplace(detId, value_LC);
      }
    }
  }

  double value_LC_prime = 1.;

  if (m_timeDependent) {
    if (detId.subdetId() != EcalSubdetector::EcalPreshower) {
      auto cache = m_valueLCCache_LC_prime.find(detId);
      if (cache != m_valueLCCache_LC_prime.end()) {
        value_LC_prime = cache->second;
      } else {
        value_LC_prime = findLaserConstant_LC_prime(detId);
        m_valueLCCache_LC_prime.emplace(detId, value_LC_prime);
      }
    }
  }

  correction_factor_for_premixed_sample_transparency = value_LC_prime / value_LC;
  //
  // LC' /  LC  (see formula)
  //

  for (int isample = 0; isample < digi.size(); ++isample) {
    int gainId = digi[isample].gainId();
    //int gainId = 1;

    if (gainId == 1) {
      result[isample] = float(digi[isample].adc()) / 1000. / peToA *
                        correction_factor_for_premixed_sample_transparency;  // special coding, save low level info
    } else if (gainId > 1) {
      result[isample] =
          float(digi[isample].adc()) * LSB[gainId - 1] * icalconst / peToA *
          correction_factor_for_premixed_sample_transparency;  // bet that no pileup hit has an energy over Emax/2
    }                                                          // gain = 0
    else {
      result[isample] =
          float(digi[isample].adc()) * LSB[gainId] * icalconst / peToA *
          correction_factor_for_premixed_sample_transparency;  //not sure we ever get here at gain=0, but hit wil be saturated anyway
      // in EcalCoder.cc it is actually "LSB[3]" -> grrr
    }

    // old version:
    //result[isample] = float(digi[isample].adc())*LSB[gainId]*icalconst/peToA;
  }

  //std::cout << " EcalSignalGenerator:EB noise input " << digi << std::endl;

  //std::cout << " converted noise sample " << std::endl;
  //for(int isample = 0; isample<digi.size(); ++isample){
  //  std::cout << " " << result[isample] ;
  //}
  //std::cout << std::endl;

  return result;
}

template <>
CaloSamples EcalSignalGenerator<EEDigitizerTraits>::samplesInPE(const DIGI &digi) {
  // calibration, for future reference:  (same block for all Ecal types)
  //EcalDetId cell = digi.id();
  //         const EcalCalibrations& calibrations=conditions->getEcalCalibrations(cell);
  //const EcalQIECoder* channelCoder = theConditions->getEcalCoder (cell);
  //const EcalQIEShape* channelShape = theConditions->getEcalShape (cell);
  //EcalCoderDb coder (*channelCoder, *channelShape);
  //CaloSamples result;
  //coder.adc2fC(digi, result);
  //fC2pe(result);

  DetId detId = digi.id();

  double Emax = fullScaleEnergy(detId);
  double LSB[NGAINS + 1];

  double icalconst = 1.;  //findIntercalibConstant( detId );

  const EcalIntercalibConstantMCMap &icalMap = ical->getMap();
  EcalIntercalibConstantMCMap::const_iterator icalit = icalMap.find(detId);
  if (icalit != icalMap.end()) {
    icalconst = (*icalit);
  }

  double peToA = peToAConversion(detId);

  const std::vector<float> gainRatios = GetGainRatios(detId);

  for (unsigned int igain(0); igain <= NGAINS; ++igain) {
    LSB[igain] = 0.;
    if (igain > 0)
      LSB[igain] = Emax / (MAXADC * gainRatios[igain]);
  }

  //    std::cout << " intercal, LSBs, egains " << icalconst << " " << LSB[0] << " " << LSB[1] << " " << gainRatios[0] << " " << gainRatios[1] << " " << Emax << std::endl;

  CaloSamples result(detId, digi.size());

  // correction facotr for premixed sample: ratio of laser corrections
  double correction_factor_for_premixed_sample_transparency = 1.0;
  double value_LC = 1.;

  if (m_timeDependent) {
    if (detId.subdetId() != EcalSubdetector::EcalPreshower) {
      auto cache = m_valueLCCache_LC.find(detId);
      if (cache != m_valueLCCache_LC.end()) {
        value_LC = cache->second;
      } else {
        value_LC = findLaserConstant_LC(detId);
        m_valueLCCache_LC.emplace(detId, value_LC);
      }
    }
  }

  double value_LC_prime = 1.;
  if (m_timeDependent) {
    if (detId.subdetId() != EcalSubdetector::EcalPreshower) {
      auto cache = m_valueLCCache_LC_prime.find(detId);
      if (cache != m_valueLCCache_LC_prime.end()) {
        value_LC_prime = cache->second;
      } else {
        value_LC_prime = findLaserConstant_LC_prime(detId);
        m_valueLCCache_LC_prime.emplace(detId, value_LC_prime);
      }
    }
  }

  correction_factor_for_premixed_sample_transparency = value_LC_prime / value_LC;
  //
  // LC' /  LC  (see formula)
  //

  for (int isample = 0; isample < digi.size(); ++isample) {
    int gainId = digi[isample].gainId();
    //int gainId = 1;

    if (gainId == 1) {
      result[isample] = float(digi[isample].adc()) / 1000. / peToA *
                        correction_factor_for_premixed_sample_transparency;  // special coding
    } else if (gainId > 1) {
      result[isample] = float(digi[isample].adc()) * LSB[gainId - 1] * icalconst / peToA *
                        correction_factor_for_premixed_sample_transparency;
    }  // gain = 0
    else {
      result[isample] = float(digi[isample].adc()) * LSB[gainId] * icalconst / peToA *
                        correction_factor_for_premixed_sample_transparency;
    }

    // old version
    //result[isample] = float(digi[isample].adc())*LSB[gainId]*icalconst/peToA;
  }

  //std::cout << " EcalSignalGenerator:EE noise input " << digi << std::endl;

  //std::cout << " converted noise sample " << std::endl;
  //for(int isample = 0; isample<digi.size(); ++isample){
  //  std::cout << " " << result[isample] ;
  // }
  //std::cout << std::endl;

  return result;
}

template <>
CaloSamples EcalSignalGenerator<ESDigitizerTraits>::samplesInPE(const DIGI &digi) {
  // calibration, for future reference:  (same block for all Ecal types)
  //EcalDetId cell = digi.id();
  //         const EcalCalibrations& calibrations=conditions->getEcalCalibrations(cell);
  //const EcalQIECoder* channelCoder = theConditions->getEcalCoder (cell);
  //const EcalQIEShape* channelShape = theConditions->getEcalShape (cell);
  //EcalCoderDb coder (*channelCoder, *channelShape);
  //CaloSamples result;
  //coder.adc2fC(digi, result);
  //fC2pe(result);

  DetId detId = digi.id();

  double icalconst = 1.;  //findIntercalibConstant( detId );

  const ESIntercalibConstantMap &icalMap = esmips->getMap();
  ESIntercalibConstantMap::const_iterator icalit = icalMap.find(detId);
  if (icalit != icalMap.end()) {
    icalconst = double(*icalit);
  }

  CaloSamples result(detId, digi.size());

  for (int isample = 0; isample < digi.size(); ++isample) {
    result[isample] = float(digi[isample].adc()) / icalconst * ESMIPToGeV;
  }

  //std::cout << " EcalSignalGenerator:ES noise input " << digi << std::endl;

  //std::cout << " converted noise sample " << std::endl;
  //for(int isample = 0; isample<digi.size(); ++isample){
  //  std::cout << " " << result[isample] ;
  //}
  //std::cout << std::endl;

  return result;
}
