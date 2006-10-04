#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitCorrection.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalHitCorrection::HcalHitCorrection(const HcalSimParameterMap * parameterMap)
: theParameterMap(parameterMap)
{
}


void HcalHitCorrection::fillChargeSums(MixCollection<PCaloHit> & hits)
{
  clear();
  for(MixCollection<PCaloHit>::MixItr hitItr = hits.begin();
      hitItr != hits.end(); ++hitItr)
  {
    int tbin = timeBin(*hitItr);
    if(tbin >= 0 && tbin < 10) 
    {  
      theChargeSumsForTimeBin[tbin][HcalDetId(hitItr->id())] += charge(*hitItr);
    }
  }
}


void HcalHitCorrection::clear()
{
  for(int i = 0; i < 10; ++i)
  {
    theChargeSumsForTimeBin[i].clear();
  }
}

double HcalHitCorrection::charge(const PCaloHit & hit) const
{
  DetId detId(hit.id());
  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
  double simHitToCharge = parameters.simHitToPhotoelectrons()
                        * parameters.photoelectronsToAnalog();

  return hit.energy() * simHitToCharge;
}


double HcalHitCorrection::delay(const PCaloHit & hit) const 
{
  // HO goes slow, HF shouldn't be used at all
  HcalDetId hcalDetId(hit.id());
  if(hcalDetId.subdet() == HcalForward) return 0;
  HcalTimeSlew::BiasSetting biasSetting = (hcalDetId.subdet() == HcalOuter) ?
                                          HcalTimeSlew::Slow :
                                          HcalTimeSlew::Medium;
  double delay = 0.;
  int tbin = timeBin(hit);
  if(tbin >= 0 && tbin < 10)
  {
    ChargeSumsByChannel::const_iterator totalChargeItr = theChargeSumsForTimeBin[tbin].find(hcalDetId);
    if(totalChargeItr == theChargeSumsForTimeBin[tbin].end())
    {
      throw cms::Exception("HcalHitCorrection") << "Cannot find HCAL charge sum for hit " << hit;
    }
    double totalCharge = totalChargeItr->second;
    delay = HcalTimeSlew::delay(totalCharge, biasSetting);
//    std::cout << "TIMESLEWcharge " << charge(hit) << "  totalcharge " << totalCharge 
//              << " olddelay " << HcalTimeSlew::delay(charge(hit), biasSetting) 
//              << " newdelay " << delay << std::endl;

  }

  return delay;
}


void HcalHitCorrection::correct(PCaloHit & hit) const {
  // replace the hit with a new one, with a time delay
  hit = PCaloHit(hit.id(), hit.energy(), hit.time()+delay(hit), hit.geantTrackId());
}


int HcalHitCorrection::timeBin(const PCaloHit & hit) const
{
  const CaloSimParameters & parameters = theParameterMap->simParameters(DetId(hit.id()));
  double t = hit.time() - timeOfFlight(HcalDetId(hit.id())) + parameters.timePhase();
  return static_cast<int> (t / 25) + parameters.binOfMaximum() - 1;
}


double HcalHitCorrection::timeOfFlight(const HcalDetId & hcalDetId) const
{
  switch(hcalDetId.subdet())
  {
  case HcalBarrel:
    return 8.4;
    break;
  case HcalEndcap:
    return 13.;
    break;
  case HcalOuter:
    return 18.7;
    break;
  case HcalForward:
    return 37.;
    break;
  default:
    throw cms::Exception("HcalHitCorrection") << "Bad Hcal subdetector " << hcalDetId.subdet();
    break;
  }
}

