#include "SimCalorimetry/HcalSimAlgos/interface/HcalTimeSlewSim.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalTimeSlewSim::HcalTimeSlewSim(const CaloVSimParameterMap * parameterMap)
  : theParameterMap(parameterMap),theRandGaussQ(0) 
{
}



double HcalTimeSlewSim::charge(const CaloSamples & samples) const
{
  double totalCharge = 0.;
  for(int i = 0; i < 4; ++i) {
    int bin = i + samples.presamples();
    if(bin < samples.size()) {
      totalCharge += samples[bin];
    }
  }
  return totalCharge;
}


void HcalTimeSlewSim::delay(CaloSamples & samples) const 
{
  // HO goes slow, HF shouldn't be used at all
  //ZDC not used for the moment

  DetId detId(samples.id());
  if(detId.det()==DetId::Calo && detId.subdetId()==HcalZDCDetId::SubdetectorId) return;
  HcalDetId hcalDetId(detId);

  if(hcalDetId.subdet() == HcalBarrel || hcalDetId.subdet() == HcalEndcap || hcalDetId.subdet() == HcalOuter ) {

    HcalTimeSlew::BiasSetting biasSetting = (hcalDetId.subdet() == HcalOuter) ?
      HcalTimeSlew::Slow :
      HcalTimeSlew::Medium;

    double totalCharge = charge(samples);
    if(totalCharge <= 0.) totalCharge = 1.e-6; // protecion against negaive v.
    double delay = HcalTimeSlew::delay(totalCharge, biasSetting);
    // now, the smearing
    const HcalSimParameters& params=static_cast<const HcalSimParameters&>(theParameterMap->simParameters(detId));
    if (params.doTimeSmear() && theRandGaussQ!=0) {
      double rms=params.timeSmearRMS(totalCharge);
      double smearns=theRandGaussQ->fire()*rms;

      LogDebug("HcalTimeSlewSim") << "TimeSmear charge " << totalCharge << " rms " << rms << " delay " << delay << " smearns " << smearns;
      delay+=smearns;
    }

    samples.offsetTime(delay);
  }
}


void HcalTimeSlewSim::setRandomEngine(CLHEP::HepRandomEngine & engine) {
  if (theRandGaussQ==0) {
    theRandGaussQ=new CLHEP::RandGaussQ(engine);
  }
}
