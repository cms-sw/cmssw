#include "SimCalorimetry/HcalSimAlgos/interface/HFSimParameters.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/HcalObjects/interface/HcalGain.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"
#include "CLHEP/Random/RandGaussQ.h"


HFSimParameters::HFSimParameters(double simHitToPhotoelectrons, double photoelectronsToAnalog, 
                                 double samplingFactor, double timePhase, bool syncPhase)
: CaloSimParameters(simHitToPhotoelectrons, photoelectronsToAnalog, samplingFactor, timePhase,
                    6, 4, false, syncPhase),
  theDbService(0),
  theSamplingFactor( samplingFactor )
{
}

HFSimParameters::HFSimParameters(const edm::ParameterSet & p)
:  CaloSimParameters(p),
   theDbService(0),
   theSamplingFactor( p.getParameter<double>("samplingFactor") )
{
}


double HFSimParameters::photoelectronsToAnalog(const DetId & detId) const
{
  // pe/fC = pe/GeV * GeV/fC  = (0.24 pe/GeV * 6 for photomult * 0.2146GeV/fC)
  return 1./(theSamplingFactor * simHitToPhotoelectrons(detId) * fCtoGeV(detId));
}

double HFSimParameters::fCtoGeV(const DetId & detId) const
{
  assert(theDbService != 0);
  HcalGenericDetId hcalGenDetId(detId);
  const HcalGain* gains = theDbService->getGain(hcalGenDetId);
  const HcalGainWidth* gwidths = theDbService->getGainWidth(hcalGenDetId);
  double result = 0.0;
  if (!gains || !gwidths )
  {
    edm::LogError("HcalAmplifier") << "Could not fetch HCAL conditions for channel " << hcalGenDetId;
  }
  else 
  {
    // only one gain will be recorded per channel, so just use capID 0 for now
    result = gains->getValue(0);
    //  if(doNoise_)
    ///  {
    //    result += CLHEP::RandGaussQ::shoot(0.,  gwidths->getValue(0));
    //  }
  }
  return result;
}


double HFSimParameters::samplingFactor() const
{
  return theSamplingFactor;
}
