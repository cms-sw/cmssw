#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
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
  HcalCalibrations calibrations;
  HcalCalibrationWidths widths;
  if (!theDbService->makeHcalCalibration (hcalDetId, &calibrations) ||
      !theDbService->makeHcalCalibrationWidth (hcalDetId, &widths)) 
  {
    edm::LogError("HcalAmplifier") << "Could not fetch HCAL conditions" ;
  }

  // the gain is in units of GeV/fC.  We want a constant with fC/pe.
  // looking at SimParameterMap, we derive that
  // fC/pe = (GeV/dGeV) / (pe/dGeV) / (GeV/fC)
  // the first two terms are the (GeV/pe)
  const CaloSimParameters & parameters = theParameterMap->simParameters(frame.id());
  double GeVperPE = parameters.samplingFactor()
                  / parameters.simHitToPhotoelectrons();

  for(int tbin = 0; tbin < frame.size(); ++tbin) {
    int capId = (theStartingCapId + tbin)%4;
    LogDebug("HcalAmplifier") << "PEDS " << capId << " " << calibrations.pedestal(capId)
        << " " << widths.pedestal(capId) << " " << calibrations.gain(capId) 
        <<" " << widths.gain(capId);
    double pedestal = calibrations.pedestal(capId);
    double gain = calibrations.gain(capId);
    if(addNoise_) {
      pedestal += RandGauss::shoot(0. , widths.pedestal(capId));
      gain += RandGauss::shoot(0., widths.gain(capId));
    }
    // since gain is (GeV/fC)
    double fCperPE = GeVperPE / gain;
    frame[tbin] *= fCperPE;
    frame[tbin] += pedestal;
  }
  LogDebug("HcalAmplifier") << frame;
}



