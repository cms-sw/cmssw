#include "SimCalorimetry/CastorSim/src/CastorSimParameters.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/CastorObjects/interface/CastorGain.h"
#include "CondFormats/CastorObjects/interface/CastorGainWidth.h"
#include "CLHEP/Random/RandGaussQ.h"


CastorSimParameters::CastorSimParameters(double simHitToPhotoelectrons, double photoelectronsToAnalog,double samplingFactor, double timePhase, bool syncPhase)
: CaloSimParameters(simHitToPhotoelectrons, photoelectronsToAnalog, samplingFactor, timePhase, 6, 4, false, syncPhase),
  theDbService(0),
  theSamplingFactor( samplingFactor ),
  nominalfCperPE( 1),
  dynamicNoise(false)
{
}


CastorSimParameters::CastorSimParameters(const edm::ParameterSet & p)
:  CaloSimParameters(p),
   theDbService(0),
   theSamplingFactor( p.getParameter<double>("samplingFactor") ),
   nominalfCperPE( p.getParameter<double>("photoelectronsToAnalog") ),
   dynamicNoise(p.getParameter<bool>("doDynamicNoise") )
{
}

double CastorSimParameters::getNominalfCperPE() const  
{
  // return the nominal PMT gain value of CASTOR from the config file.
  return nominalfCperPE;
}

bool CastorSimParameters::doDynamicNoise() const  
{
  // activate proper noise treatment depending on used gain values.
  return dynamicNoise;
}

double CastorSimParameters::photoelectronsToAnalog(const DetId & detId) const
{
  // calculate factor (PMT gain) using sampling factor value & available electron gain
  return theSamplingFactor/fCtoGeV(detId);
}



double CastorSimParameters::fCtoGeV(const DetId & detId) const
{
  assert(theDbService != nullptr);
  HcalGenericDetId hcalGenDetId(detId);
  const CastorGain* gains = theDbService->getGain(hcalGenDetId);
  const CastorGainWidth* gwidths = theDbService->getGainWidth(hcalGenDetId);
  double result = 0.0;
  if (!gains || !gwidths )
  {
    edm::LogError("CastorAmplifier") << "Could not fetch HCAL conditions for channel " << hcalGenDetId;
  }
  else 
  {
    // only one gain will be recorded per channel, so just use capID 0 for now
    result = gains->getValue(0);
  }
  return result;
}
