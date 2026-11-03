#include "SimCalorimetry/EcalSimAlgos/interface/EcalLiteDTUCoder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "DataFormats/EcalDigi/interface/EcalLiteDTUSample.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"

#include <algorithm>
#include <cmath>

EcalLiteDTUCoder::EcalLiteDTUCoder(bool addNoise,
                                   bool PreMix1,
                                   unsigned int lowGainPreSamples,
                                   unsigned int lowGainPostSamples,
                                   EcalLiteDTUCoder::Noisifier* ebCorrNoise0,
                                   EcalLiteDTUCoder::Noisifier* ebCorrNoise1)
    : m_peds(nullptr),
      m_gainRatios(0),
      m_intercals(nullptr),
      m_maxEneEB(ecalPh2::maxEneEB),  // Maximum for CATIA: LSB gain 10: 0.048 MeV
      m_addNoise(addNoise),
      m_PreMix1(PreMix1),
      m_lowGainPreSamples(lowGainPreSamples),
      m_lowGainPostSamples(lowGainPostSamples),
      m_ebCorrNoise{ebCorrNoise0, ebCorrNoise1} {}

void EcalLiteDTUCoder::setFullScaleEnergy(double EBscale) { m_maxEneEB = EBscale; }

void EcalLiteDTUCoder::setPedestals(const EcalLiteDTUPedestalsMap* pedestals) { m_peds = pedestals; }

void EcalLiteDTUCoder::setGainRatios(float gainRatios) { m_gainRatios = gainRatios; }

void EcalLiteDTUCoder::setIntercalibConstants(const EcalIntercalibConstantsMC* ical) { m_intercals = ical; }

void EcalLiteDTUCoder::analogToDigital(CLHEP::HepRandomEngine* engine,
                                       const EcalSamples& clf,
                                       EcalDataFrame_Ph2& df) const {
  df.setSize(clf.size());
  encode(clf, df, engine);
}

void EcalLiteDTUCoder::encode(const EcalSamples& ecalSamples,
                              EcalDataFrame_Ph2& df,
                              CLHEP::HepRandomEngine* engine) const {
  auto const nSamples = static_cast<int>(ecalSamples.size());

  auto const detId = ecalSamples.id();

  std::array<double, ecalPh2::NGAINS> pedestals;
  std::array<double, ecalPh2::NGAINS> widths;
  std::array<double, ecalPh2::NGAINS> LSB;
  std::array<double, ecalPh2::NGAINS> trueRMS;
  double icalconst = 1.;
  findIntercalibConstant(detId, icalconst);

  for (unsigned int igain(0); igain < ecalPh2::NGAINS; ++igain) {
    // fill in the pedestal and width
    findPedestal(detId, igain, pedestals[igain], widths[igain]);
    // insert an absolute value in the trueRMS
    trueRMS[igain] = std::sqrt(std::abs(widths[igain] * widths[igain] - 1. / 12.));

    LSB[igain] = m_maxEneEB / (ecalPh2::MAXADC * ecalPh2::gains[igain]);
  }

  std::array<CaloSamples, ecalPh2::NGAINS> noiseframe = {
      CaloSamples(detId, nSamples),
      CaloSamples(detId, nSamples),
  };

  if (m_addNoise) {
    for (unsigned int ig = 0; ig < ecalPh2::NGAINS; ++ig) {
      m_ebCorrNoise[ig]->noisify(noiseframe[ig], engine);
    }
  }

  std::vector<std::array<int, ecalPh2::NGAINS>> adctrace(nSamples, {-1, -1});
  int firstSaturatedSampleG10 = -1;
  int lastSaturatedSampleG10 = -1;

  // fill ADC trace in gain 0 (x10) and gain 1 (x1)
  for (unsigned int igain = 0; igain < ecalPh2::NGAINS; ++igain) {
    for (int i(0); i != nSamples; ++i) {
      double asignal = 0;
      if (!m_PreMix1) {
        asignal = pedestals[igain] + ecalSamples[i] / (LSB[igain] * icalconst) + trueRMS[igain] * noiseframe[igain][i];
        // Analog signal value for each sample in ADC.
        // It is corrected by the intercalibration constants
      } else {
        // no noise nor pedestal when premixing
        asignal = ecalSamples[i] / (LSB[igain] * icalconst);
      }

      auto const adc = std::round(asignal);
      // gain 0 (x10) channel is saturated, readout will use gain 1 (x10), but I count the number of saturated samples
      if (igain == 0 && adc > ecalPh2::MAXADC) {
        if (firstSaturatedSampleG10 < 0)
          firstSaturatedSampleG10 = i;
        lastSaturatedSampleG10 = i;
      }
      // the Lite-DTU ADC works in binary mode and does not return negative values
      adctrace[i][igain] = static_cast<int>(std::clamp(adc, 0., static_cast<double>(ecalPh2::MAXADC)));
    }
    if (firstSaturatedSampleG10 < 0) {
      break;  // gain 0 (x10) is not saturated, so don't bother with gain 1
    }
  }  // for igain

  // Limits of gain 1:
  // The Lite DTU sends a number of samples before the saturating one, and a number of sample after the last with gain 1.
  // This is configurable and a typical setting is 2 pre samples and 5 post sample to have at least 8 low gain sample in case one sample is saturated.
  // The assumption is that the saturated samples are consecutive.
  auto const firstLowGainSample = firstSaturatedSampleG10 - static_cast<int>(m_lowGainPreSamples);
  auto const lastLowGainSample = lastSaturatedSampleG10 + static_cast<int>(m_lowGainPostSamples);

  // Setting values to the samples:
  for (int j = 0; j < nSamples; ++j) {
    int const igain = firstSaturatedSampleG10 > -1 and j >= firstLowGainSample and j <= lastLowGainSample ? 1 : 0;
    df.setSample(j, EcalLiteDTUSample(adctrace[j][igain], igain));
  }
}

void EcalLiteDTUCoder::findPedestal(const DetId& detId, int gainId, double& ped, double& width) const {
  EcalLiteDTUPedestalsMap::const_iterator itped = m_peds->getMap().find(detId);
  if (itped != m_peds->getMap().end()) {
    ped = (*itped).mean(gainId);
    width = (*itped).rms(gainId);
    LogDebug("EcalLiteDTUCoder") << "Pedestals for " << detId.rawId() << " gain range " << gainId << ": "
                                 << "Mean = " << ped << " rms = " << width;
  } else {
    LogDebug("EcalLiteDTUCoder") << "Pedestals not found, put default values (ped: 12; width: 2.5)";
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
    LogDebug("EcalLiteDTUCoder") << "Intercalibration constant not found. Using default value (1.).";
    thisconst = 1.;
  }

  if (icalconst == 0.)
    thisconst = 1.;

  icalconst = thisconst;
}
