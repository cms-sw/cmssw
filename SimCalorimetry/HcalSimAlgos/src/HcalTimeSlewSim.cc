#include "SimCalorimetry/HcalSimAlgos/interface/HcalTimeSlewSim.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandGaussQ.h"

HcalTimeSlewSim::HcalTimeSlewSim(const CaloVSimParameterMap * parameterMap)
  : theParameterMap(parameterMap)
{
}


// not quite adequate to 25ns high-PU regime
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


void HcalTimeSlewSim::delay(CaloSamples & samples, CLHEP::HepRandomEngine* engine) const
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

    // double totalCharge = charge(samples); 

    int maxbin =  samples.size();
    CaloSamples data(detId, maxbin);   // for a temporary copy 
    data =  samples;  

    for(int i = 0; i < samples.size()-1; ++i) {
      double totalCharge = data[i]/0.6;   // temporary change from total charge to approximation TS/0.6
                                          // until we get more precise/reliable QIE8 simulation  

      if(totalCharge <= 0.) totalCharge = 1.e-6; // protecion against negaive v.
      double delay = HcalTimeSlew::delay(totalCharge, biasSetting);
      // now, the smearing still remains
      const HcalSimParameters& params=static_cast<const HcalSimParameters&>(theParameterMap->simParameters(detId));
      if (params.doTimeSmear()) {
	double rms=params.timeSmearRMS(totalCharge);
	double smearns=CLHEP::RandGaussQ::shoot(engine)*rms;
	
	LogDebug("HcalTimeSlewSim") << "TimeSmear charge " << totalCharge << " rms " << rms << " delay " << delay << " smearns " << smearns;
	delay+=smearns;
      }
      
      // samples.offsetTime(delay);  -> replacing it with 1TS move 

      double t = i*25. - delay;
      int firstbin = floor(t/25.);
      double f = t/25. - firstbin;
      int nextbin = firstbin + 1;
      double v2 = (nextbin < 0  || nextbin  >= maxbin) ? 0. : data[nextbin];
      data[i] = v2*f;
      data[i+1] = data[i+1] + (v2 - data[i]); 
    }
    samples = data;
  }
}
