#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
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

HcalAmplifier::HcalAmplifier(const CaloVSimParameterMap * parameters, bool addNoise) :
  theDbService(0), 
  theParameterMap(parameters),
  theStartingCapId(0), 
  addNoise_(addNoise)
{
}


void HcalAmplifier::amplify(CaloSamples & frame) const {
  assert(theDbService != 0);
  HcalDetId hcalDetId(frame.id());
  const HcalPedestal* peds = theDbService->getPedestal  (hcalDetId);
  const HcalGain* gains = theDbService->getGain  (hcalDetId);
  const HcalPedestalWidth* pwidths = theDbService->getPedestalWidth  (hcalDetId);
  const HcalGainWidth* gwidths = theDbService->getGainWidth  (hcalDetId);
  if (!peds || !gains || !pwidths || !gwidths )
  {
    edm::LogError("HcalAmplifier") << "Could not fetch HCAL conditions for channel " << hcalDetId;
  }

  // the gain is in units of GeV/fC.  We want a constant with fC/pe.
  // looking at SimParameterMap, we derive that
  // fC/pe = (GeV/dGeV) / (pe/dGeV) / (GeV/fC)
  // the first two terms are the (GeV/pe)
  const CaloSimParameters & parameters = theParameterMap->simParameters(frame.id());
  double GeVperPE = parameters.samplingFactor()
                  / parameters.simHitToPhotoelectrons();

  double gauss [32]; //big enough
  double noise [32]; //big enough
  for (int i = 0; i < frame.size(); i++) gauss[i] = RandGauss::shoot(0., 1.);
  pwidths->makeNoise (frame.size(), gauss, noise);
  for(int tbin = 0; tbin < frame.size(); ++tbin) {
    int capId = (theStartingCapId + tbin)%4;
    LogDebug("HcalAmplifier") << "PEDS " << capId << " " << peds->getValue (capId)
        << " " << pwidths->getWidth (capId) << " " << gains->getValue (capId)
        << " " << gwidths->getValue (capId);
    double pedestal = peds->getValue (capId);
    double gain = gains->getValue (capId);
    if(addNoise_) {
      pedestal += noise [tbin];
      gain += RandGauss::shoot(0., gwidths->getValue (capId));
    }
    // since gain is (GeV/fC)
    double fCperPE = GeVperPE / gain;
    frame[tbin] *= fCperPE;
    frame[tbin] += pedestal;
  }
  LogDebug("HcalAmplifier") << frame;
}



