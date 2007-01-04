#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/HcalObjects/interface/HcalGain.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"
#include "CLHEP/Random/RandGaussQ.h"


HcalSimParameters::HcalSimParameters(double simHitToPhotoelectrons, double photoelectronsToAnalog,
                 double samplingFactor, double timePhase,
                 int readoutFrameSize, int binOfMaximum,
                 bool doPhotostatistics, bool syncPhase,
                 int firstRing, const std::vector<double> & samplingFactors)
: CaloSimParameters(simHitToPhotoelectrons, photoelectronsToAnalog, samplingFactor, timePhase,
                    readoutFrameSize, binOfMaximum, doPhotostatistics, syncPhase),
  theDbService(0),
  theFirstRing(firstRing),
  theSamplingFactors(samplingFactors)
{
}



/*
HcalSimParameters::HcalSimParameters(const edm::ParameterSet & p)
:  CaloSimParameters(p),
   theDbService(0),
   theFirstRing( p.getParameter<int>("firstRing") ),
   theSamplingFactors( p.getParameter<std::vector<double> >("samplingFactors") )
{
}
*/

double HcalSimParameters::simHitToPhotoelectrons(const DetId & detId) const 
{
  // the gain is in units of GeV/fC.  We want a constant with pe/dGeV
  // pe/dGeV = (GeV/dGeV) / (GeV/fC) / (fC/pe)
  
  return samplingFactor(detId) / fCtoGeV(detId) / photoelectronsToAnalog();
}


double HcalSimParameters::fCtoGeV(const DetId & detId) const
{
  assert(theDbService != 0);
  HcalDetId hcalDetId(detId);
  const HcalGain* gains = theDbService->getGain  (hcalDetId);
  const HcalGainWidth* gwidths = theDbService->getGainWidth  (hcalDetId);
  if (!gains || !gwidths )
  {
    edm::LogError("HcalAmplifier") << "Could not fetch HCAL conditions for channel " << hcalDetId;
  }
  // only one gain will be recorded per channel, so just use capID 0 for now
  
  double result = gains->getValue(0);
//  if(doNoise_)
///  {
//    result += RandGaussQ::shoot(0.,  gwidths->getValue(0));
//  }
  return result;
}

/*
double HcalSimParameters::photoelectronsToAnalog(const DetId & detId) const
{ 
  // multiply amplification factor by electron charge to get fC
  return amplification_ * (1.6e-4);
}
*/

double HcalSimParameters::samplingFactor(const DetId & detId) const
{
  HcalDetId hcalDetId(detId);
  return theSamplingFactors[hcalDetId.ietaAbs()-theFirstRing];
}

