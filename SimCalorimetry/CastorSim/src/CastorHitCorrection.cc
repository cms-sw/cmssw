#include "SimCalorimetry/CastorSim/src/CastorHitCorrection.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorTimeSlew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

CastorHitCorrection::CastorHitCorrection(const CaloVSimParameterMap * parameterMap)
: theParameterMap(parameterMap)
{
}


void CastorHitCorrection::fillChargeSums(MixCollection<PCaloHit> & hits)
{
  //  clear();
  for(MixCollection<PCaloHit>::MixItr hitItr = hits.begin();
      hitItr != hits.end(); ++hitItr)
  {
    LogDebug("CastorHitCorrection") << "CastorHitCorrection::Hit 0x" << std::hex << hitItr->id() << std::dec;
    int tbin = timeBin(*hitItr);
    LogDebug("CastorHitCorrection") << "CastorHitCorrection::Hit tbin" << tbin;
    if(tbin >= 0 && tbin < 10) 
    {  
      theChargeSumsForTimeBin[tbin][DetId(hitItr->id())] += charge(*hitItr);
    }
  }
}

void CastorHitCorrection::fillChargeSums(const std::vector<PCaloHit> & hits)
{
  //  clear();
  for(std::vector<PCaloHit>::const_iterator hitItr = hits.begin();
      hitItr != hits.end(); ++hitItr)
  {
    LogDebug("CastorHitCorrection") << "CastorHitCorrection::Hit 0x" << std::hex << hitItr->id() << std::dec;
    int tbin = timeBin(*hitItr);
    LogDebug("CastorHitCorrection") << "CastorHitCorrection::Hit tbin" << tbin;
    if(tbin >= 0 && tbin < 10) 
    {  
      theChargeSumsForTimeBin[tbin][DetId(hitItr->id())] += charge(*hitItr);
    }
  }
}


void CastorHitCorrection::clear()
{
  for(int i = 0; i < 10; ++i)
  {
    theChargeSumsForTimeBin[i].clear();
  }
}

double CastorHitCorrection::charge(const PCaloHit & hit) const
{
  DetId detId(hit.id());
  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
  double simHitToCharge = parameters.simHitToPhotoelectrons()
                        * parameters.photoelectronsToAnalog();
  return hit.energy() * simHitToCharge;
}

double CastorHitCorrection::delay(const PCaloHit & hit) const 
{
  // HO goes slow, HF shouldn't be used at all
  //Castor not used for the moment

  DetId detId(hit.id());
  if(detId.det()==DetId::Calo && (detId.subdetId()==HcalCastorDetId::SubdetectorId)) return 0;

  HcalDetId hcalDetId(hit.id());
  if(hcalDetId.subdet() == HcalForward) return 0;  
  CastorTimeSlew::BiasSetting biasSetting = (hcalDetId.subdet() == HcalOuter) ?
                                          CastorTimeSlew::Slow :
                                          CastorTimeSlew::Medium;
  double delay = 0.;
  int tbin = timeBin(hit);
  if(tbin >= 0 && tbin < 10)
  {
    ChargeSumsByChannel::const_iterator totalChargeItr = theChargeSumsForTimeBin[tbin].find(detId);
    if(totalChargeItr == theChargeSumsForTimeBin[tbin].end())
    {
      throw cms::Exception("CastorHitCorrection") << "Cannot find HCAL/CASTOR charge sum for hit " << hit;
    }
    double totalCharge = totalChargeItr->second;
    delay = CastorTimeSlew::delay(totalCharge, biasSetting);
    LogDebug("CastorHitCorrection") << "TIMESLEWcharge " << charge(hit) 
				  << "  totalcharge " << totalCharge 
				  << " olddelay "  << CastorTimeSlew::delay(charge(hit), biasSetting) 
				  << " newdelay " << delay;
  }

  return delay;
}


int CastorHitCorrection::timeBin(const PCaloHit & hit) const
{
  const CaloSimParameters & parameters = theParameterMap->simParameters(DetId(hit.id()));
  double t = hit.time() - timeOfFlight(DetId(hit.id())) + parameters.timePhase();
  return static_cast<int> (t / 25) + parameters.binOfMaximum() - 1;
}


double CastorHitCorrection::timeOfFlight(const DetId & detId) const
{
    if(detId.det()==DetId::Calo && detId.subdetId()==HcalCastorDetId::SubdetectorId)
	return 37.666;
    else
	throw cms::Exception("not HcalCastorDetId"); 
}

