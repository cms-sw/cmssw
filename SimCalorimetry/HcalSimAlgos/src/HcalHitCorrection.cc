#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitCorrection.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalHitCorrection::HcalHitCorrection(const HcalSimParameterMap * parameterMap)
: theParameterMap(parameterMap)
{
}


void HcalHitCorrection::correct(PCaloHit & hit) const {
  DetId detId(hit.id());
  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
  double simHitToCharge = parameters.simHitToPhotoelectrons()
                        * parameters.photoelectronsToAnalog();

  double charge = hit.energy() * simHitToCharge;

  // HO goes slow, HF shouldn't be used at all
  HcalSubdetector subdet = HcalDetId(hit.id()).subdet();
  assert(subdet != HcalForward);
  HcalTimeSlew::BiasSetting biasSetting = (subdet == HcalOuter) ? 
                                          HcalTimeSlew::Slow :
                                          HcalTimeSlew::Medium;

  double delay = HcalTimeSlew::delay(charge, biasSetting);

  LogDebug("HcalHitCorrection") << "Hcal time slew of " << delay 
     << " ns for signal of " << charge; 

  // replace the hit with a new one, with a time delay
  hit = PCaloHit(hit.id(), hit.energy(), hit.time()+delay, hit.geantTrackId());
}


