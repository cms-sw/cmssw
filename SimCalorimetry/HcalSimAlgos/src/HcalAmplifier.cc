#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDIonFeedbackSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTimeSlewSim.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseSignalGenerator.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalGain.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandFlat.h"

#include <cmath>
#include <cmath>

HcalAmplifier::HcalAmplifier(const CaloVSimParameterMap* parameters, bool addNoise, bool PreMix1, bool PreMix2)
    : theDbService(nullptr),
      theParameterMap(parameters),
      theNoiseSignalGenerator(nullptr),
      theIonFeedbackSim(nullptr),
      theTimeSlewSim(nullptr),
      theStartingCapId(0),
      addNoise_(addNoise),
      preMixDigi_(PreMix1),
      preMixAdd_(PreMix2) {}

void HcalAmplifier::setDbService(const HcalDbService* service) {
  theDbService = service;
  if (theIonFeedbackSim)
    theIonFeedbackSim->setDbService(service);
}

void HcalAmplifier::amplify(CaloSamples& frame, CLHEP::HepRandomEngine* engine) const {
  if (theIonFeedbackSim) {
    theIonFeedbackSim->addThermalNoise(frame, engine);
  }
  pe2fC(frame);

  if (frame.id().det() == DetId::Hcal && ((frame.id().subdetId() == HcalGenericDetId::HcalGenBarrel) ||
                                          (frame.id().subdetId() == HcalGenericDetId::HcalGenEndcap))) {
    const HcalSimParameters& params = static_cast<const HcalSimParameters&>(theParameterMap->simParameters(frame.id()));
    if (abs(params.delayQIE()) <= 25)
      applyQIEdelay(frame, params.delayQIE());
  }

  // don't bother for blank signals
  if (theTimeSlewSim && frame.size() > 4 && frame[4] > 1.e-6) {
    theTimeSlewSim->delay(frame, engine, theTimeSlew);
  }

  // if we are combining pre-mixed digis, we need noise and peds
  if (theNoiseSignalGenerator == nullptr || preMixAdd_ || !theNoiseSignalGenerator->contains(frame.id())) {
    addPedestals(frame, engine);
  }
  LogDebug("HcalAmplifier") << frame;
}

void HcalAmplifier::pe2fC(CaloSamples& frame) const {
  const CaloSimParameters& parameters = theParameterMap->simParameters(frame.id());
  frame *= parameters.photoelectronsToAnalog(frame.id());
}

void HcalAmplifier::applyQIEdelay(CaloSamples& cs, int delayQIE) const {
  DetId detId(cs.id());
  int maxbin = cs.size();
  int precisebin = cs.preciseSize();
  CaloSamples data(detId, maxbin, precisebin);  // make a temporary copy
  data = cs;
  data.setBlank();
  data.resetPrecise();

  for (int i = 0; i < precisebin; i++) {
    int j = i + 2 * delayQIE;
    data.preciseAtMod(i) +=
        // value positive (signal moves earlier in time)
        delayQIE > 0 ? (j < precisebin ? cs.preciseAt(j) : 0.) :
                     //  value = 0 or negative (signal gets delayed)
            (j < 0 ? 0. : cs.preciseAt(j));

    int samplebin = (int)i * maxbin / precisebin;
    data[samplebin] += data.preciseAt(i);
  }

  cs = data;  // update the sample
}

void HcalAmplifier::addPedestals(CaloSamples& frame, CLHEP::HepRandomEngine* engine) const {
  assert(theDbService != nullptr);
  HcalGenericDetId hcalGenDetId(frame.id());
  HcalGenericDetId::HcalGenericSubdetector hcalSubDet = hcalGenDetId.genericSubdet();

  if (!((frame.id().subdetId() == HcalGenericDetId::HcalGenBarrel) ||
        (frame.id().subdetId() == HcalGenericDetId::HcalGenEndcap) ||
        (frame.id().subdetId() == HcalGenericDetId::HcalGenForward) ||
        (frame.id().subdetId() == HcalGenericDetId::HcalGenOuter)))
    return;

  if (hcalGenDetId.isHcalCastorDetId())
    return;
  if (hcalGenDetId.isHcalZDCDetId())
    return;

  const HcalCalibrationWidths& calibWidths = theDbService->getHcalCalibrationWidths(hcalGenDetId);
  const HcalCalibrations& calibs = theDbService->getHcalCalibrations(hcalGenDetId);

  double noise[32] = {0.};  //big enough
  if (addNoise_) {
    double gauss[32];  //big enough
    for (int i = 0; i < frame.size(); i++)
      gauss[i] = CLHEP::RandGaussQ::shoot(engine, 0., 1.);
    makeNoise(hcalSubDet, calibWidths, frame.size(), gauss, noise);
  }

  if (!preMixDigi_) {  // if we are doing initial premix, no pedestals
    for (int tbin = 0; tbin < frame.size(); ++tbin) {
      int capId = (theStartingCapId + tbin) % 4;
      double pedestal = calibs.pedestal(capId) + noise[tbin];
      frame[tbin] += pedestal;
    }
  }
}

void HcalAmplifier::makeNoise(HcalGenericDetId::HcalGenericSubdetector hcalSubDet,
                              const HcalCalibrationWidths& width,
                              int fFrames,
                              double* fGauss,
                              double* fNoise) const {
  // This is a simplified noise generation scheme using only the diagonal elements
  // (proposed by Salavat Abduline).
  // This is direct adaptation of the code in HcalPedestalWidth.cc

  // average over capId's
  double s_xx_mean = 0.25 * (width.pedestal(0) * width.pedestal(0) + width.pedestal(1) * width.pedestal(1) +
                             width.pedestal(2) * width.pedestal(2) + width.pedestal(3) * width.pedestal(3));

  // Off-diagonal element approximation
  // In principle should come from averaging the values of elements (0.1), (1,2), (2,3), (3,0)
  // For now use the definition below (but keep structure of the code structure for development)
  double s_xy_mean = -0.5 * s_xx_mean;
  // Use different parameter for HF to reproduce the noise rate after zero suppression.
  // Steven Won/Jim Hirschauer/Radek Ofierzynski 18.03.2010
  if (hcalSubDet == HcalGenericDetId::HcalGenForward)
    s_xy_mean = 0.08 * s_xx_mean;

  double term = s_xx_mean * s_xx_mean - 2. * s_xy_mean * s_xy_mean;

  if (term < 0.)
    term = 1.e-50;
  double sigma = sqrt(0.5 * (s_xx_mean + sqrt(term)));
  double corr = sigma == 0. ? 0. : 0.5 * s_xy_mean / sigma;

  for (int i = 0; i < fFrames; i++) {
    fNoise[i] = fGauss[i] * sigma;
    if (i > 0)
      fNoise[i] += fGauss[i - 1] * corr;
    if (i < fFrames - 1)
      fNoise[i] += fGauss[i + 1] * corr;
  }
}
