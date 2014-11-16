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
#include <math.h>

HcalAmplifier::HcalAmplifier(const CaloVSimParameterMap * parameters, bool addNoise, bool PreMix1, bool PreMix2) :
  theDbService(nullptr),
  theParameterMap(parameters),
  theNoiseSignalGenerator(nullptr),
  myCholeskys(nullptr),
  myADCPeds(nullptr),
  theIonFeedbackSim(nullptr),
  theTimeSlewSim(nullptr),
  theStartingCapId(0),
  addNoise_(addNoise),
  preMixDigi_(PreMix1),
  preMixAdd_(PreMix2),
  useOldHB(false),
  useOldHE(false),
  useOldHF(false),
  useOldHO(false)
{ }


void HcalAmplifier::setDbService(const HcalDbService * service) {
  theDbService = service;
  if(theIonFeedbackSim) theIonFeedbackSim->setDbService(service);
}


void HcalAmplifier::amplify(CaloSamples & frame, CLHEP::HepRandomEngine* engine) const {
  if(theIonFeedbackSim)
  {
    theIonFeedbackSim->addThermalNoise(frame, engine);
  }
  pe2fC(frame);
  // don't bother for blank signals
  if(theTimeSlewSim && frame[4] > 1.e-6)
  {
    theTimeSlewSim->delay(frame, engine);
  }

  // if we are combining pre-mixed digis, we need noise and peds
  if(theNoiseSignalGenerator==0 || preMixAdd_ || !theNoiseSignalGenerator->contains(frame.id()) )
  {
    addPedestals(frame, engine);
  }
  LogDebug("HcalAmplifier") << frame;
}


void HcalAmplifier::pe2fC(CaloSamples & frame) const
{
  const CaloSimParameters & parameters = theParameterMap->simParameters(frame.id());
  frame *= parameters.photoelectronsToAnalog(frame.id());
}

void HcalAmplifier::setHBtuningParameter(double tp) { HB_ff = tp; }
void HcalAmplifier::setHEtuningParameter(double tp) { HE_ff = tp; }
void HcalAmplifier::setHFtuningParameter(double tp) { HF_ff = tp; }
void HcalAmplifier::setHOtuningParameter(double tp) { HO_ff = tp; }
void HcalAmplifier::setUseOldHB(bool useOld) { useOldHB = useOld; }
void HcalAmplifier::setUseOldHE(bool useOld) { useOldHE = useOld; }
void HcalAmplifier::setUseOldHF(bool useOld) { useOldHF = useOld; }
void HcalAmplifier::setUseOldHO(bool useOld) { useOldHO = useOld; }

void HcalAmplifier::addPedestals(CaloSamples & frame, CLHEP::HepRandomEngine* engine) const
{
   assert(theDbService != 0);
   HcalGenericDetId hcalGenDetId(frame.id());
   HcalGenericDetId::HcalGenericSubdetector hcalSubDet = hcalGenDetId.genericSubdet();

   bool useOld=false;
   if(hcalSubDet==HcalGenericDetId::HcalGenBarrel) useOld = useOldHB;
   if(hcalSubDet==HcalGenericDetId::HcalGenEndcap) useOld = useOldHE;
   if(hcalSubDet==HcalGenericDetId::HcalGenForward) useOld = useOldHF;
   if(hcalSubDet==HcalGenericDetId::HcalGenOuter) useOld = useOldHO;
   
   if(useOld)
   {
     const HcalCalibrationWidths & calibWidths =
       theDbService->getHcalCalibrationWidths(hcalGenDetId);
     const HcalCalibrations& calibs = theDbService->getHcalCalibrations(hcalGenDetId);
   
     double noise [32] = {0.}; //big enough
     if(addNoise_)
     {
       double gauss [32]; //big enough
       for (int i = 0; i < frame.size(); i++) gauss[i] = CLHEP::RandGaussQ::shoot(engine, 0., 1.);
       makeNoiseOld(hcalSubDet, calibWidths, frame.size(), gauss, noise);
     }
   
     if(!preMixDigi_){  // if we are doing initial premix, no pedestals
       for (int tbin = 0; tbin < frame.size(); ++tbin) {
	 int capId = (theStartingCapId + tbin)%4;
	 double pedestal = calibs.pedestal(capId) + noise[tbin];

	 frame[tbin] += pedestal;
       }
     }
     return;
   }


  double fudgefactor = 1;
  if(hcalSubDet==HcalGenericDetId::HcalGenBarrel) fudgefactor = HB_ff;
  if(hcalSubDet==HcalGenericDetId::HcalGenEndcap) fudgefactor = HE_ff;
  if(hcalSubDet==HcalGenericDetId::HcalGenForward) fudgefactor = HF_ff;
  if(hcalSubDet==HcalGenericDetId::HcalGenOuter) fudgefactor = HO_ff;

  if ( !( (frame.id().subdetId()==HcalGenericDetId::HcalGenBarrel) ||
	  (frame.id().subdetId()==HcalGenericDetId::HcalGenEndcap) ||
	  (frame.id().subdetId()==HcalGenericDetId::HcalGenForward) ||
	  (frame.id().subdetId()==HcalGenericDetId::HcalGenOuter) ) ) return;

  if(hcalGenDetId.isHcalCastorDetId()) return;
  if(hcalGenDetId.isHcalZDCDetId()) return;

  int theStartingCapId_2 = (int)floor(CLHEP::RandFlat::shoot(engine, 0., 4.));
  double noise [32] = {0.}; //big enough

  if( myCholeskys ) {
    const HcalCholeskyMatrix * thisChanCholesky = myCholeskys->getValues(hcalGenDetId,false);
    if ( !thisChanCholesky ) {
      edm::LogWarning("HcalAmplifier") << "no Cholesky " << hcalSubDet << " "
				       << hcalGenDetId.rawId() << " " 
				       << frame.id().subdetId();
      return;
    }

    if(addNoise_)
    {
      double gauss [32]; //big enough
      for (int i = 0; i < frame.size(); i++) gauss[i] = CLHEP::RandGaussQ::shoot(engine, 0., 1.);
      makeNoise(*thisChanCholesky, frame.size(), gauss, noise, (int)theStartingCapId_2);
    }
  } else if(addNoise_) {
    /* TODO: Re-add the Cholesky matrix when computing the noise.
     * The payloads currently stored in Condition DB suffer a bug in ROOT5 streamer.
     * The workaround here is to fall back to the old noise computation when the matrix is not provided.
     * When a permanent fix is found, this condition should be removed:
     * when Cholesky matrix is not set from the digitizer, issue a warning.
     * -Salvatore Di Guida
     */
    const HcalCalibrationWidths & calibWidths = theDbService->getHcalCalibrationWidths(hcalGenDetId);
    double gauss [32]; //big enough
    for (int i = 0; i < frame.size(); i++) gauss[i] = CLHEP::RandGaussQ::shoot(engine, 0., 1.);
    makeNoiseOld(hcalSubDet, calibWidths, frame.size(), gauss, noise);
  } else {
    edm::LogWarning("HcalAmplifier") << "No Cholesky Matrices provided for new HCAL noise simulation.";
  }
  
  if( myADCPeds ) {
    const HcalPedestal* thisChanADCPeds = myADCPeds->getValues(hcalGenDetId);
    const HcalQIECoder* coder = theDbService->getHcalCoder(hcalGenDetId);
    const HcalQIEShape* shape = theDbService->getHcalShape(coder);

    for (int tbin = 0; tbin < frame.size(); ++tbin) {
      int capId = (theStartingCapId_2 + tbin)%4;
      double x = noise[tbin] * fudgefactor + thisChanADCPeds->getValue(capId);//*(values+capId); //*.70 goes here!
      int x1=(int)std::floor(x);
      int x2=(int)std::floor(x+1);
      float y2=coder->charge(*shape,x2,capId);
      float y1=coder->charge(*shape,x1,capId);
      frame[tbin] = (y2-y1)*(x-x1)+y1;
    }
  } else {
    edm::LogWarning("HcalAmplifier") << "No ADC pedestals provided for new HCAL simulation.";
  }
}

void HcalAmplifier::makeNoise (const HcalCholeskyMatrix & thisChanCholesky, int fFrames, double* fGauss, double* fNoise, int m) const {
   if(fFrames > 10) return;

   for(int i = 0; i != 10; i++){
      for(int j = 0; j != 10; j++){ //fNoise is initialized to zero in function above! Must be zero before this step
         fNoise[i] += thisChanCholesky.getValue(m,i,j) * fGauss[j];
      }
   }
}

void HcalAmplifier::makeNoiseOld (HcalGenericDetId::HcalGenericSubdetector hcalSubDet, const HcalCalibrationWidths& width, int fFrames, double* fGauss, double* fNoise) const 
{
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
  // Use different parameter for HF to reproduce the noise rate after zero suppression.
  // Steven Won/Jim Hirschauer/Radek Ofierzynski 18.03.2010
  if (hcalSubDet == HcalGenericDetId::HcalGenForward) s_xy_mean = 0.08 * s_xx_mean;

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

