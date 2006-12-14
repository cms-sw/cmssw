#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalGain.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Random/RandGaussQ.h"

#include<iostream>

HcalAmplifier::HcalAmplifier(const HcalSimParameterMap * parameters, bool addNoise) :
  theDbService(0), 
  theParameterMap(parameters),
  theStartingCapId(0), 
  addNoise_(addNoise)
{
}


void HcalAmplifier::amplify(CaloSamples & frame) const {
  const CaloSimParameters & parameters = theParameterMap->simParameters(frame.id());
  assert(theDbService != 0);
  HcalDetId hcalDetId(frame.id());
  const HcalPedestal* peds = theDbService->getPedestal  (hcalDetId);
  const HcalPedestalWidth* pwidths = theDbService->getPedestalWidth  (hcalDetId);
  if (!peds || !pwidths )
  {
    edm::LogError("HcalAmplifier") << "Could not fetch HCAL conditions for channel " << hcalDetId;
  }

  double gauss [32]; //big enough
  double noise [32]; //big enough
  double fCperPE = parameters.photoelectronsToAnalog();


  for (int i = 0; i < frame.size(); i++) gauss[i] = RandGaussQ::shoot(0., 1.);
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
  LogDebug("HcalAmplifier") << frame;
}

