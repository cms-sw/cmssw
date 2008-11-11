#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalGain.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

#include<iostream>
#include <math.h>

HcalAmplifier::HcalAmplifier(const CaloVSimParameterMap * parameters, bool addNoise) :
  theDbService(0), 
  theRandGaussQ(0),
  theParameterMap(parameters),
  theStartingCapId(0), 
  addNoise_(addNoise)
{
}


void HcalAmplifier::setRandomEngine(CLHEP::HepRandomEngine & engine)
{
  theRandGaussQ = new CLHEP::RandGaussQ(engine);
}

void HcalAmplifier::amplify(CaloSamples & frame) const {
  const CaloSimParameters & parameters = theParameterMap->simParameters(frame.id());
  assert(theDbService != 0);
  HcalGenericDetId hcalGenDetId(frame.id());

  // Access in HcalDbService was changed
  //  HcalCalibrationWidths calibWidths;
  //  theDbService->makeHcalCalibrationWidth(hcalGenDetId, &calibWidths);

  const HcalCalibrationWidths& calibWidths = theDbService->getHcalCalibrationWidths(hcalGenDetId);
  const HcalCalibrations& calibs = theDbService->getHcalCalibrations(hcalGenDetId);

  /*
  // checks performed when using the "old" inputs 
  if (!peds || !pwidths )
  {
    edm::LogError("HcalAmplifier") << "Could not fetch HCAL conditions for channel " << hcalGenDetId;
  }
  */

  double gauss [32]; //big enough
  double noise [32]; //big enough
  double fCperPE = parameters.photoelectronsToAnalog(frame.id());

  for (int i = 0; i < frame.size(); i++) gauss[i] = theRandGaussQ->fire(0., 1.);

  makeNoise(calibWidths, frame.size(), gauss, noise);

  for(int tbin = 0; tbin < frame.size(); ++tbin) {
    int capId = (theStartingCapId + tbin)%4;

    double pedestal = calibs.pedestal(capId);

    if(addNoise_) {
      pedestal += noise [tbin];
    }
    frame[tbin] *= fCperPE;
    frame[tbin] += pedestal;
  }
  LogDebug("HcalAmplifier") << frame;
}




void HcalAmplifier::makeNoise (const HcalCalibrationWidths& width, int fFrames, double* fGauss, double* fNoise) const {

  // This is a simplified noise generation scheme using only the diagonal elements
  // (proposed by Salavat Abduline).
  // This is direct adaptation of the code in HcalPedestalWidth.cc
  
  // average over capId's
  double s_xx_mean =  0.25 * (width.pedestal(0)*width.pedestal(0) + 
			      width.pedestal(1)*width.pedestal(1) + 
			      width.pedestal(2)*width.pedestal(2) + 
			      width.pedestal(3)*width.pedestal(3));


  // Off-diagonal element approximation
  // In principle should come from averaging the values of elements (0.1), (1,2), (2,3), (3,0)
  // For now use the definition below (but keep structure of the code structure for development) 
  double s_xy_mean = -0.5 * s_xx_mean;

  double term  = s_xx_mean*s_xx_mean - 2.*s_xy_mean*s_xy_mean;

  if (term < 0.) term = 1.e-50 ;
  double sigma = sqrt (0.5 * (s_xx_mean + sqrt(term)));
  double corr = sigma == 0. ? 0. : 0.5*s_xy_mean / sigma;

  for (int i = 0; i < fFrames; i++) {
    fNoise [i] = fGauss[i]*sigma;
    if (i > 0) fNoise [i] += fGauss[i-1]*corr;
    if (i < fFrames-1) fNoise [i] += fGauss[i+1]*corr;
  }
}

