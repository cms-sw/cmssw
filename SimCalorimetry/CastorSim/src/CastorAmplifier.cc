#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CondFormats/CastorObjects/interface/CastorGain.h"
#include "CondFormats/CastorObjects/interface/CastorGainWidth.h"
#include "CondFormats/CastorObjects/interface/CastorPedestal.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidth.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimCalorimetry/CastorSim/src/CastorAmplifier.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameters.h"

#include "CLHEP/Random/RandGaussQ.h"

#include <iostream>
#include <cassert>

CastorAmplifier::CastorAmplifier(const CastorSimParameterMap *parameters, bool addNoise)
    : theDbService(nullptr), theParameterMap(parameters), theStartingCapId(0), addNoise_(addNoise) {}

void CastorAmplifier::amplify(CaloSamples &frame, CLHEP::HepRandomEngine *engine) const {
  const CastorSimParameters &parameters = theParameterMap->castorParameters();
  assert(theDbService != nullptr);
  HcalGenericDetId hcalGenDetId(frame.id());
  const CastorPedestal *peds = theDbService->getPedestal(hcalGenDetId);
  const CastorPedestalWidth *pwidths = theDbService->getPedestalWidth(hcalGenDetId);
  if (!peds || !pwidths) {
    edm::LogError("CastorAmplifier") << "Could not fetch HCAL/CASTOR conditions for channel " << hcalGenDetId;
  } else {
    double gauss[32];  // big enough
    double noise[32];  // big enough
    double fCperPE = parameters.photoelectronsToAnalog(frame.id());
    double nominalfCperPE = parameters.getNominalfCperPE();

    for (int i = 0; i < frame.size(); i++) {
      gauss[i] = CLHEP::RandGaussQ::shoot(engine, 0., 1.);
    }
    if (addNoise_) {
      pwidths->makeNoise(frame.size(), gauss, noise);
    }
    for (int tbin = 0; tbin < frame.size(); ++tbin) {
      int capId = (theStartingCapId + tbin) % 4;
      double pedestal = peds->getValue(capId);
      if (addNoise_) {
        pedestal += noise[tbin] * (fCperPE / nominalfCperPE);
      }
      frame[tbin] *= fCperPE;
      frame[tbin] += pedestal;
    }
  }
  LogDebug("CastorAmplifier") << frame;
}
