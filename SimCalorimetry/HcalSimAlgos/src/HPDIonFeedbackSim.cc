
// --------------------------------------------------------
// A class to simulated HPD ion feedback noise.
// The deliverable of the class is the ion feedback noise
// for an HcalDetId units of fC or GeV
//
// Project: HPD ion feedback
// Author: T.Yetkin University of Iowa, Feb. 16, 2010
// --------------------------------------------------------

#include "SimCalorimetry/HcalSimAlgos/interface/HPDIonFeedbackSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "CondFormats/HcalObjects/interface/HcalGain.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include "CLHEP/Random/RandBinomial.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"

using namespace edm;
using namespace std;

// constants for simulation/parameterization
static const double pe2Charge = 0.333333;  // fC/p.e.

HPDIonFeedbackSim::HPDIonFeedbackSim(const edm::ParameterSet& iConfig, const CaloShapes* shapes)
    : theDbService(nullptr), theShapes(shapes) {}

HPDIonFeedbackSim::~HPDIonFeedbackSim() {}

double HPDIonFeedbackSim::getIonFeedback(
    DetId detId, double signal, double pedWidth, bool doThermal, bool isInGeV, CLHEP::HepRandomEngine* engine) {
  //    HcalDetId id = detId;

  double GeVperfC = 1.;
  if (isInGeV)
    GeVperfC = 1. / fCtoGeV(detId);

  double charge = signal / GeVperfC;

  double noise = 0.;             // fC
  if (charge > 3. * pedWidth) {  // 3 sigma away from pedestal mean
    int npe = int(charge / pe2Charge);
    if (doThermal) {
      double electronEmission = 0.08;
      CLHEP::RandPoissonQ theRandPoissonQ(*engine, electronEmission);
      npe += theRandPoissonQ.fire();
    }

    noise = correctPE(detId, npe, engine) - npe;
  }
  return (noise * GeVperfC);
}

double HPDIonFeedbackSim::correctPE(const DetId& detId, double npe, CLHEP::HepRandomEngine* engine) const {
  double rateInTail = 0.000211988;        //read this from XML file
  double rateInSecondTail = 4.61579e-06;  //read this from XML file

  // three gauss fit is applied to data to get ion feedback distribution
  // parameters (in fC)
  // first gaussian
  // double p0 = 9.53192e+05;
  // double p1 = -3.13653e-01;
  // double p2 = 2.78350e+00;

  // second gaussian
  // double p3 = 2.41611e+03;
  double p4 = 2.06117e+01;
  double p5 = 1.09239e+01;

  // third gaussian
  // double p6 = 3.42793e+01;
  double p7 = 5.45548e+01;
  double p8 = 1.59696e+01;

  double noise = 0.;  // fC
  int nFirst = (int)(CLHEP::RandBinomial::shoot(engine, npe, rateInTail));
  int nSecond = (int)(CLHEP::RandBinomial::shoot(engine, npe, rateInSecondTail));

  for (int j = 0; j < nFirst; ++j) {
    noise += CLHEP::RandGaussQ::shoot(engine, p4, p5);
  }
  for (int j = 0; j < nSecond; ++j) {
    noise += CLHEP::RandGaussQ::shoot(engine, p7, p8);
  }

  return npe + std::max(noise / pe2Charge, 0.);
}

void HPDIonFeedbackSim::addThermalNoise(CaloSamples& samples, CLHEP::HepRandomEngine* engine) {
  // make some chance to add a PE (with a chance of feedback)
  // for each time sample
  double meanPE = 0.02;
  DetId detId(samples.id());
  int nSamples = samples.size();
  const CaloVShape* shape = theShapes->shape(detId);
  for (int i = 0; i < nSamples; ++i) {
    CLHEP::RandPoissonQ theRandPoissonQ(*engine, meanPE);
    double npe = theRandPoissonQ.fire();
    // TODOprobably should time-smear these
    if (npe > 0.) {
      // chance of feedback
      npe = correctPE(detId, npe, engine);
      for (int j = i; j < nSamples; ++j) {
        double timeFromPE = (j - i) * 25.;
        samples[j] += (*shape)(timeFromPE)*npe;
      }
    }
  }
}

double HPDIonFeedbackSim::fCtoGeV(const DetId& detId) const {
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
  }
  return result;
}
