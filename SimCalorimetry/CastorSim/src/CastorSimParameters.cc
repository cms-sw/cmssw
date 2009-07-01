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
  theSamplingFactor( samplingFactor )
{
}

/* 
CastorSimParameters::CastorSimParameters(double simHitToPhotoelectrons, double photoelectronsToAnalog,
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
*/

CastorSimParameters::CastorSimParameters(const edm::ParameterSet & p)
:  CaloSimParameters(p),
   theDbService(0),
   theSamplingFactor( p.getParameter<double>("samplingFactor") )
{
}
/*
: CaloSimParameters(p),
  theDbService(0),
  theFirstRing( p.getParameter<int>("firstRing") ),
  theSamplingFactors( p.getParameter<std::vector<double> >("samplingFactors") )
{
}
*/

/*
double CastorSimParameters::simHitToPhotoelectrons(const DetId & detId) const 
{
  // the gain is in units of GeV/fC.  We want a constant with pe/dGeV
  // pe/dGeV = (GeV/dGeV) / (GeV/fC) / (fC/pe) 
  double result = CaloSimParameters::simHitToPhotoelectrons(detId);
  if(HcalGenericDetId(detId).genericSubdet() != HcalGenericDetId::HcalGenForward
     || HcalGenericDetId(detId).genericSubdet() != HcalGenericDetId::HcalGenCastor)
    { 
      result = samplingFactor(detId) / fCtoGeV(detId) / photoelectronsToAnalog();
    }
  return result;
}
*/

double CastorSimParameters::photoelectronsToAnalog(const DetId & detId) const
{
  // pe/fC = pe/GeV * GeV/fC  = (0.24 pe/GeV * 6 for photomult * 0.2146GeV/fC)
  return 1./(theSamplingFactor * simHitToPhotoelectrons(detId) * fCtoGeV(detId));
}



double CastorSimParameters::fCtoGeV(const DetId & detId) const
{
  assert(theDbService != 0);
  HcalGenericDetId hcalGenDetId(detId);
  const CastorGain* gains = theDbService->getGain(hcalGenDetId);
  const CastorGainWidth* gwidths = theDbService->getGainWidth(hcalGenDetId);
  if (!gains || !gwidths )
  {
    edm::LogError("CastorAmplifier") << "Could not fetch HCAL conditions for channel " << hcalGenDetId;
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
double CastorSimParameters::samplingFactor(const DetId & detId) const {  
HcalDetId hcalDetId(detId); 
return theSamplingFactors[hcalDetId.ietaAbs()-theFirstRing];
}
*/
