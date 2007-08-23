#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
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

HcalSimParameters::HcalSimParameters(const edm::ParameterSet & p)
:  CaloSimParameters(p),
   theDbService(0),
   theFirstRing( p.getParameter<int>("firstRing") ),
   theSamplingFactors( p.getParameter<std::vector<double> >("samplingFactors") )
{
}

double HcalSimParameters::simHitToPhotoelectrons(const DetId & detId) const 
{
  // the gain is in units of GeV/fC.  We want a constant with pe/dGeV
  // pe/dGeV = (GeV/dGeV) / (GeV/fC) / (fC/pe) 
  double result = CaloSimParameters::simHitToPhotoelectrons(detId);
  if(HcalGenericDetId(detId).genericSubdet() != HcalGenericDetId::HcalGenForward
     || HcalGenericDetId(detId).genericSubdet() != HcalGenericDetId::HcalGenZDC)
    { 
      result = samplingFactor(detId) / fCtoGeV(detId) / photoelectronsToAnalog();
    }
  return result;
}


double HcalSimParameters::fCtoGeV(const DetId & detId) const
{
  assert(theDbService != 0);
  HcalGenericDetId hcalGenDetId(detId);
  const HcalGain* gains = theDbService->getGain(hcalGenDetId);
  const HcalGainWidth* gwidths = theDbService->getGainWidth(hcalGenDetId);
  if (!gains || !gwidths )
  {
    edm::LogError("HcalAmplifier") << "Could not fetch HCAL conditions for channel " << hcalGenDetId;
  }
  // only one gain will be recorded per channel, so just use capID 0 for now
  
  double result = gains->getValue(0);
//  if(doNoise_)
///  {
//    result += RandGaussQ::shoot(0.,  gwidths->getValue(0));
//  }
  return result;
}

double HcalSimParameters::samplingFactor(const DetId & detId) const
{
  HcalDetId hcalDetId(detId);
  return theSamplingFactors[hcalDetId.ietaAbs()-theFirstRing];
}

