#include "SimCalorimetry/CastorSim/src/CastorAmplifier.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameters.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CondFormats/CastorObjects/interface/CastorPedestal.h"
#include "CondFormats/CastorObjects/interface/CastorGain.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidth.h"
#include "CondFormats/CastorObjects/interface/CastorGainWidth.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>

CastorAmplifier::CastorAmplifier(const CastorSimParameterMap * parameters, bool addNoise) :
  theDbService(0), 
  theRandGaussQ(0),
  theParameterMap(parameters),
  theStartingCapId(0), 
  addNoise_(addNoise)
{
}


void CastorAmplifier::setRandomEngine(CLHEP::HepRandomEngine & engine)
{
  theRandGaussQ = new CLHEP::RandGaussQ(engine);
}

void CastorAmplifier::amplify(CaloSamples & frame) const {
  const CastorSimParameters & parameters = theParameterMap->castorParameters();
  assert(theDbService != 0);
  HcalGenericDetId hcalGenDetId(frame.id());
  const CastorPedestal* peds = theDbService->getPedestal(hcalGenDetId);
  const CastorPedestalWidth* pwidths = theDbService->getPedestalWidth(hcalGenDetId);
  if (!peds || !pwidths )
  {
    edm::LogError("CastorAmplifier") << "Could not fetch HCAL/CASTOR conditions for channel " << hcalGenDetId;
  }

  double gauss [32]; //big enough
  double noise [32]; //big enough
  double fCperPE = parameters.photoelectronsToAnalog(frame.id());

  for (int i = 0; i < frame.size(); i++) gauss[i] = theRandGaussQ->fire(0., 1.);
  pwidths->makeNoise (frame.size(), gauss, noise);
  for(int tbin = 0; tbin < frame.size(); ++tbin) {
    int capId = (theStartingCapId + tbin)%4;
    double pedestal = peds->getValue (capId);
    if(addNoise_) {
      pedestal += noise [tbin];
    }
    frame[tbin] *= fCperPE;
    frame[tbin] += pedestal;
  }
  LogDebug("CastorAmplifier") << frame;
}

