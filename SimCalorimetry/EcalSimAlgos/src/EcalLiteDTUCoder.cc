#include "SimCalorimetry/EcalSimAlgos/interface/EcalLiteDTUCoder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "DataFormats/EcalDigi/interface/EcalLiteDTUSample.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"

#include <iostream>

EcalLiteDTUCoder::EcalLiteDTUCoder(bool addNoise,
                                   bool PreMix1,
                                   EcalLiteDTUCoder::Noisifier* ebCorrNoise0,
                                   EcalLiteDTUCoder::Noisifier* ebCorrNoise1)
    : m_peds(nullptr),
      m_gainRatios(0),
      m_intercals(nullptr),
      m_maxEneEB(ecalPh2::maxEneEB),  // Maximum for CATIA: LSB gain 10: 0.048 MeV
      m_addNoise(addNoise),
      m_PreMix1(PreMix1),
      m_ebCorrNoise{ebCorrNoise0, ebCorrNoise1}

{}

EcalLiteDTUCoder::~EcalLiteDTUCoder() {}

void EcalLiteDTUCoder::setFullScaleEnergy(double EBscale) { m_maxEneEB = ecalPh2::maxEneEB; }

void EcalLiteDTUCoder::setPedestals(const EcalLiteDTUPedestalsMap* pedestals) { m_peds = pedestals; }

void EcalLiteDTUCoder::setGainRatios(float gainRatios) { m_gainRatios = gainRatios; }

void EcalLiteDTUCoder::setIntercalibConstants(const EcalIntercalibConstantsMC* ical) { m_intercals = ical; }

double EcalLiteDTUCoder::fullScaleEnergy(const DetId& detId) const { return m_maxEneEB; }

void EcalLiteDTUCoder::analogToDigital(CLHEP::HepRandomEngine* engine,
                                       const EcalSamples& clf,
                                       EcalDataFrame_Ph2& df) const {
  df.setSize(clf.size());
  encode(clf, df, engine);
}

void EcalLiteDTUCoder::encode(const EcalSamples& ecalSamples,
                              EcalDataFrame_Ph2& df,
                              CLHEP::HepRandomEngine* engine) const {
  const int nSamples(ecalSamples.size());

  DetId detId = ecalSamples.id();
  double Emax = fullScaleEnergy(detId);

  //N Gains set to 2 in the .h
  double pedestals[ecalPh2::NGAINS];
  double widths[ecalPh2::NGAINS];
  double LSB[ecalPh2::NGAINS];
  double trueRMS[ecalPh2::NGAINS];
  int nSaturatedSamples = 0;
  double icalconst = 1.;
  findIntercalibConstant(detId, icalconst);

  for (unsigned int igain(0); igain < ecalPh2::NGAINS; ++igain) {
    // fill in the pedestal and width
    findPedestal(detId, igain, pedestals[igain], widths[igain]);
    // insert an absolute value in the trueRMS
    trueRMS[igain] = std::sqrt(std::abs(widths[igain] * widths[igain] - 1. / 12.));

    LSB[igain] = Emax / (ecalPh2::MAXADC * ecalPh2::gains[igain]);
  }

  CaloSamples noiseframe[ecalPh2::NGAINS] = {
      CaloSamples(detId, nSamples),
      CaloSamples(detId, nSamples),
  };

  const Noisifier* noisy[ecalPh2::NGAINS] = {m_ebCorrNoise[0], m_ebCorrNoise[1]};

  if (m_addNoise) {
    for (unsigned int ig = 0; ig < ecalPh2::NGAINS; ++ig) {
      noisy[ig]->noisify(noiseframe[ig], engine);
    }
  }

  std::vector<std::vector<int>> adctrace(nSamples);
  int firstSaturatedSample[ecalPh2::NGAINS] = {0, 0};

  for (int i(0); i != nSamples; ++i)
    adctrace[i].resize(ecalPh2::NGAINS);

  for (unsigned int igain = 0; igain < ecalPh2::NGAINS; ++igain) {
    for (int i(0); i != nSamples; ++i) {
      adctrace[i][igain] = -1;
    }
  }

  // fill ADC trace in gain 0 (x10) and gain 1 (x1)
  for (unsigned int igain = 0; igain < ecalPh2::NGAINS; ++igain) {
    for (int i(0); i != nSamples; ++i) {
      double asignal = 0;
      if (!m_PreMix1) {
        asignal = pedestals[igain] + ecalSamples[i] / (LSB[igain] * icalconst) + trueRMS[igain] * noiseframe[igain][i];
        //Analog signal value for each sample in ADC.
        //It is corrected by the intercalibration constants

      } else {
        //  no noise nor pedestal when premixing
        asignal = ecalSamples[i] / (LSB[igain] * icalconst);
      }
      int isignal = asignal;

      unsigned int adc = asignal - (double)isignal < 0.5 ? isignal : isignal + 1;
      // gain 0 (x10) channel is saturated, readout will use gain 1 (x10), but I count the number of saturated samples
      if (adc > ecalPh2::MAXADC) {
        adc = ecalPh2::MAXADC;
        if (nSaturatedSamples == 0)
          firstSaturatedSample[igain] = i;
        nSaturatedSamples++;
      }
      adctrace[i][igain] = adc;
    }
    if (nSaturatedSamples == 0) {
      break;  //  gain 0 (x10) is not saturated, so don't bother with gain 1
    }
  }  // for igain

  int igain = 0;

  //Limits of gain 1:
  //The Lite DTU sends 5 samples before the saturating one, and 10 after with gain 1.
  //we put the maximum in bin 5, but could happen that the system saturates before.

  int previousSaturatedSamples = 5;
  int nextSaturatedSamples = 10;
  int startingLowerGainSample = 0;
  int endingLowerGainSample = (firstSaturatedSample[0] + nextSaturatedSamples + (nSaturatedSamples));

  if (nSaturatedSamples != 0 and (firstSaturatedSample[0] - previousSaturatedSamples) < 0) {
    startingLowerGainSample = 0;
  } else {
    startingLowerGainSample = (firstSaturatedSample[0] - previousSaturatedSamples);
  }

  //Setting values to the samples:
  for (int j = 0; j < nSamples; ++j) {
    if (nSaturatedSamples != 0 and j >= startingLowerGainSample and j < endingLowerGainSample) {
      igain = 1;
    } else {
      igain = 0;
    }
    df.setSample(j, EcalLiteDTUSample(adctrace[j][igain], igain));
  }
}

void EcalLiteDTUCoder::findPedestal(const DetId& detId, int gainId, double& ped, double& width) const {
  EcalLiteDTUPedestalsMap::const_iterator itped = m_peds->getMap().find(detId);
  if (itped != m_peds->getMap().end()) {
    ped = (*itped).mean(gainId);
    width = (*itped).rms(gainId);
    LogDebug("EcalLiteDTUCoder") << "Pedestals for " << detId.rawId() << " gain range " << gainId << " : \n"
                                 << "Mean = " << ped << " rms = " << width;
  } else {
    LogDebug("EcalLiteDTUCoder") << "Pedestals not found, put default values (ped: 12; width: 2.5) \n";
    ped = 12.;
    width = 2.5;
  }
}

void EcalLiteDTUCoder::findIntercalibConstant(const DetId& detId, double& icalconst) const {
  EcalIntercalibConstantMC thisconst = 1.;
  // find intercalib constant for this xtal
  const EcalIntercalibConstantMCMap& icalMap = m_intercals->getMap();
  EcalIntercalibConstantMCMap::const_iterator icalit = icalMap.find(detId);
  if (icalit != icalMap.end()) {
    thisconst = (*icalit);
  } else {
    LogDebug("EcalLiteDTUCoder") << "Intercalib Constant not found, put default value \n";
    thisconst = 1.;
  }

  if (icalconst == 0.)
    thisconst = 1.;

  icalconst = thisconst;
}
