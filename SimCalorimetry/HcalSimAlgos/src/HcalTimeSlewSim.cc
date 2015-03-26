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

    // smearing
    double eps = 1.e-6;
    double scale_factor = 0.6;  
    double scale = data[4] / scale_factor;      
    double smearns = 0.;

    const HcalSimParameters& params =
      static_cast<const HcalSimParameters&>(theParameterMap->simParameters(detId));
    if (params.doTimeSmear()) {
      double rms = params.timeSmearRMS(scale);
      smearns = CLHEP::RandGaussQ::shoot(engine)*rms;
      LogDebug("HcalTimeSlewSim") << "TimeSmear charge " 
				  << scale << " rms " << rms 
				  << " smearns " << smearns;
    }
    
    for(int i = 0; i < samples.size()-1; ++i) {
      double totalCharge = data[i]/scale_factor;   
      // until we get more precise/reliable QIE8 simulation...  

      double delay = smearns;
      if(totalCharge <= 0.) totalCharge = eps; // protecion against negaive v.
      delay += HcalTimeSlew::delay(totalCharge, biasSetting);      
      if(delay <= 0.) delay = eps;

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
