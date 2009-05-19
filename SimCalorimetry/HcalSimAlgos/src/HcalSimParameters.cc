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
  theSamplingFactors(samplingFactors),
  doTimeSmear_(true)
{
  defaultTimeSmearing();
}

HcalSimParameters::HcalSimParameters(const edm::ParameterSet & p)
:  CaloSimParameters(p),
   theDbService(0),
   theFirstRing( p.getParameter<int>("firstRing") ),
   theSamplingFactors( p.getParameter<std::vector<double> >("samplingFactors") ),
   doTimeSmear_( p.getParameter<bool>("timeSmearing"))
{
  defaultTimeSmearing();
}

double HcalSimParameters::simHitToPhotoelectrons(const DetId & detId) const 
{
  // the gain is in units of GeV/fC.  We want a constant with pe/dGeV
  // pe/dGeV = (GeV/dGeV) / (GeV/fC) / (fC/pe) 
  double result = CaloSimParameters::simHitToPhotoelectrons(detId);
  if(HcalGenericDetId(detId).genericSubdet() != HcalGenericDetId::HcalGenForward
     || HcalGenericDetId(detId).genericSubdet() != HcalGenericDetId::HcalGenZDC)
    { 
      result = samplingFactor(detId) / fCtoGeV(detId) / photoelectronsToAnalog(detId);
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
  return theSamplingFactors.at(hcalDetId.ietaAbs()-theFirstRing);
}

void HcalSimParameters::defaultTimeSmearing() {
  // ampl (fC), time (ns) (used 0.145 GeV/fC for conversion)
  theSmearSettings.push_back(std::pair<double,double>(27.6	, 2.2));
  theSmearSettings.push_back(std::pair<double,double>(137.9	, 2.2));
  theSmearSettings.push_back(std::pair<double,double>(172.4	,1.950));
  theSmearSettings.push_back(std::pair<double,double>(206.9	,1.809));
  theSmearSettings.push_back(std::pair<double,double>(255.2	,1.664));
  theSmearSettings.push_back(std::pair<double,double>(306.9	,1.519));
  theSmearSettings.push_back(std::pair<double,double>(386.2	,1.425));
  theSmearSettings.push_back(std::pair<double,double>(437.9	,1.348));
  theSmearSettings.push_back(std::pair<double,double>(558.6	,1.192));
  theSmearSettings.push_back(std::pair<double,double>(610.3	,1.130));
  theSmearSettings.push_back(std::pair<double,double>(789.7	,1.144));
  theSmearSettings.push_back(std::pair<double,double>(1210.3	,1.070));
  theSmearSettings.push_back(std::pair<double,double>(2413.8	,1.043));
  theSmearSettings.push_back(std::pair<double,double>(99999.00, 1.043));
}

double HcalSimParameters::timeSmearRMS(double ampl) const {
  HcalTimeSmearSettings::size_type i;
  double smearsigma=0;

  for (i=0; i<theSmearSettings.size(); i++)
    if (theSmearSettings[i].first > ampl)
      break;

  // Smearing occurs only within the envelope definitions.
  if (i!=0 && (i < theSmearSettings.size())) {
    double energy1 = theSmearSettings[i-1].first;
    double sigma1  = theSmearSettings[i-1].second;
    double energy2 = theSmearSettings[i].first;
    double sigma2  = theSmearSettings[i].second;
    
    if (energy2 != energy1)
      smearsigma = sigma1 + ((sigma2-sigma1)*(ampl-energy1)/(energy2-energy1));
    else
      smearsigma = (sigma2+sigma1)/2.;    
  }

  return smearsigma;

}
