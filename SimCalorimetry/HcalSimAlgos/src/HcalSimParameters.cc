#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/HcalObjects/interface/HcalGain.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"
#include "CalibFormats/HcalObjects/interface/HcalSiPMType.h"
#include "CLHEP/Random/RandGaussQ.h"
using namespace std;

HcalSimParameters::HcalSimParameters(double simHitToPhotoelectrons,
                                     double samplingFactor,
                                     double timePhase,
                                     int readoutFrameSize,
                                     int binOfMaximum,
                                     bool doPhotostatistics,
                                     bool syncPhase,
                                     int firstRing,
                                     const std::vector<double>& samplingFactors,
                                     double sipmTau)
    : CaloSimParameters(simHitToPhotoelectrons,
                        0.0,
                        samplingFactor,
                        timePhase,
                        readoutFrameSize,
                        binOfMaximum,
                        doPhotostatistics,
                        syncPhase),
      theDbService(nullptr),
      theSiPMcharacteristics(nullptr),
      theFirstRing(firstRing),
      theSamplingFactors(samplingFactors),
      theSiPMSmearing(false),
      doTimeSmear_(true),
      theSiPMTau(sipmTau) {
  defaultTimeSmearing();

  edm::LogInfo("HcalSimParameters:") << " doSiPMsmearing    = " << theSiPMSmearing;
}

HcalSimParameters::HcalSimParameters(const edm::ParameterSet& p)
    : CaloSimParameters(p, true),
      theDbService(nullptr),
      theFirstRing(p.getParameter<int>("firstRing")),
      theSamplingFactors(p.getParameter<std::vector<double> >("samplingFactors")),
      theSiPMSmearing(p.getParameter<bool>("doSiPMSmearing")),
      doTimeSmear_(p.getParameter<bool>("timeSmearing")),
      theSiPMTau(p.getParameter<double>("sipmTau")),
      threshold_currentTDC_(p.getParameter<double>("threshold_currentTDC")) {
  defaultTimeSmearing();

  edm::LogInfo("HcalSimParameters:") << " doSiPMsmearing    = " << theSiPMSmearing;
}

void HcalSimParameters::setDbService(const HcalDbService* service) {
  assert(service);
  theDbService = service;
  theSiPMcharacteristics = service->getHcalSiPMCharacteristics();
  assert(theSiPMcharacteristics);
}

double HcalSimParameters::simHitToPhotoelectrons(const DetId& detId) const {
  // the gain is in units of GeV/fC.  We want a constant with pe/dGeV
  // pe/dGeV = (GeV/dGeV) / (GeV/fC) / (fC/pe)
  double result = CaloSimParameters::simHitToPhotoelectrons(detId);
  if (HcalGenericDetId(detId).genericSubdet() != HcalGenericDetId::HcalGenForward ||
      HcalGenericDetId(detId).genericSubdet() != HcalGenericDetId::HcalGenZDC) {
    result = samplingFactor(detId) / fCtoGeV(detId) / photoelectronsToAnalog(detId);
  }
  return result;
}

double HcalSimParameters::fCtoGeV(const DetId& detId) const {
  assert(theDbService != nullptr);
  HcalGenericDetId hcalGenDetId(detId);
  const HcalGain* gains = theDbService->getGain(hcalGenDetId);
  const HcalGainWidth* gwidths = theDbService->getGainWidth(hcalGenDetId);
  double result = 0.0;
  if (!gains || !gwidths) {
    edm::LogError("HcalAmplifier") << "Could not fetch HCAL conditions for channel " << hcalGenDetId;
  } else {
    // only one gain will be recorded per channel, so just use capID 0 for now
    result = gains->getValue(0);
    //  if(doNoise_)
    ///  {
    //    result += CLHEP::RandGaussQ::shoot(0.,  gwidths->getValue(0));
    //  }
  }
  return result;
}

double HcalSimParameters::samplingFactor(const DetId& detId) const {
  HcalDetId hcalDetId(detId);
  return theSamplingFactors.at(hcalDetId.ietaAbs() - theFirstRing);
}

double HcalSimParameters::photoelectronsToAnalog(const DetId& detId) const {
  //now always taken from database for HPDs or SiPMs (HB, HE, HO)
  assert(theDbService);
  return theDbService->getHcalSiPMParameter(detId)->getFCByPE();
}

//static const double GeV2fC = 1.0/0.145;
static const double GeV2fC = 1.0 / 0.4;

void HcalSimParameters::defaultTimeSmearing() {
  // GeV->ampl (fC), time (ns)
  theSmearSettings.emplace_back(4.00 * GeV2fC, 4.050);
  theSmearSettings.emplace_back(20.00 * GeV2fC, 3.300);
  theSmearSettings.emplace_back(25.00 * GeV2fC, 2.925);
  theSmearSettings.emplace_back(30.00 * GeV2fC, 2.714);
  theSmearSettings.emplace_back(37.00 * GeV2fC, 2.496);
  theSmearSettings.emplace_back(44.50 * GeV2fC, 2.278);
  theSmearSettings.emplace_back(56.00 * GeV2fC, 2.138);
  theSmearSettings.emplace_back(63.50 * GeV2fC, 2.022);
  theSmearSettings.emplace_back(81.00 * GeV2fC, 1.788);
  theSmearSettings.emplace_back(88.50 * GeV2fC, 1.695);
  theSmearSettings.emplace_back(114.50 * GeV2fC, 1.716);
  theSmearSettings.emplace_back(175.50 * GeV2fC, 1.070);
  theSmearSettings.emplace_back(350.00 * GeV2fC, 1.564);
  theSmearSettings.emplace_back(99999.00 * GeV2fC, 1.564);
}

double HcalSimParameters::timeSmearRMS(double ampl) const {
  HcalTimeSmearSettings::size_type i;
  double smearsigma = 0;

  for (i = 0; i < theSmearSettings.size(); i++)
    if (theSmearSettings[i].first > ampl)
      break;

  // Smearing occurs only within the envelope definitions.
  if (i != 0 && (i < theSmearSettings.size())) {
    double energy1 = theSmearSettings[i - 1].first;
    double sigma1 = theSmearSettings[i - 1].second;
    double energy2 = theSmearSettings[i].first;
    double sigma2 = theSmearSettings[i].second;

    if (energy2 != energy1)
      smearsigma = sigma1 + ((sigma2 - sigma1) * (ampl - energy1) / (energy2 - energy1));
    else
      smearsigma = (sigma2 + sigma1) / 2.;
  }

  return smearsigma;
}

int HcalSimParameters::pixels(const DetId& detId) const {
  assert(theDbService);
  int type = theDbService->getHcalSiPMParameter(detId)->getType();
  return theSiPMcharacteristics->getPixels(type);
}

double HcalSimParameters::sipmDarkCurrentuA(const DetId& detId) const {
  assert(theDbService);
  return theDbService->getHcalSiPMParameter(detId)->getDarkCurrent();
}

double HcalSimParameters::sipmCrossTalk(const DetId& detId) const {
  assert(theDbService);
  int type = theDbService->getHcalSiPMParameter(detId)->getType();
  return theSiPMcharacteristics->getCrossTalk(type);
}
std::vector<float> HcalSimParameters::sipmNonlinearity(const DetId& detId) const {
  assert(theDbService);
  int type = theDbService->getHcalSiPMParameter(detId)->getType();
  return theSiPMcharacteristics->getNonLinearities(type);
}

unsigned int HcalSimParameters::signalShape(const DetId& detId) const {
  assert(theDbService);
  return theDbService->getHcalMCParam(detId)->signalShape();
}
