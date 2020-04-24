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

HcalTimeSlewSim::HcalTimeSlewSim(const CaloVSimParameterMap * parameterMap, double minFCToDelay)
  : theParameterMap(parameterMap),
    minFCToDelay_(minFCToDelay)
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


void HcalTimeSlewSim::delay(CaloSamples & cs, CLHEP::HepRandomEngine* engine) const
{
  // HO goes slow, HF shouldn't be used at all
  //ZDC not used for the moment

  DetId detId(cs.id());
  if(detId.det()==DetId::Calo && detId.subdetId()==HcalZDCDetId::SubdetectorId) return;
  HcalDetId hcalDetId(detId);

  if(hcalDetId.subdet() == HcalBarrel || hcalDetId.subdet() == HcalEndcap || hcalDetId.subdet() == HcalOuter ) {

    HcalTimeSlew::BiasSetting biasSetting = (hcalDetId.subdet() == HcalOuter) ?
      HcalTimeSlew::Slow :
      HcalTimeSlew::Medium;

    // double totalCharge = charge(cs); // old TS... 

    int maxbin = cs.size();
    CaloSamples data(detId, maxbin);   // for a temporary copy 
    data = cs;  

    // smearing
    int    soi          = cs.presamples();
    double eps          = 1.e-6;
    double scale_factor = 0.6;  
    double scale        = cs[soi] / scale_factor;      
    double smearns      = 0.;
    double cut          = minFCToDelay_; //5. fC (above mean) for signal to be delayed

    const HcalSimParameters& params =
      static_cast<const HcalSimParameters&>(theParameterMap->simParameters(detId));
    if (params.doTimeSmear()) {
      double rms = params.timeSmearRMS(scale);
      smearns = CLHEP::RandGaussQ::shoot(engine)*rms;
      LogDebug("HcalTimeSlewSim") << "TimeSmear charge " 
				  << scale << " rms " << rms 
				  << " smearns " << smearns;
    }

    // cycle over TS',  it - current TS index
    for(int it = 0; it < maxbin-1; ) {
 
      double datai = cs[it];
      int    nts  = 0;
      double tshift = smearns;
      double totalCharge = datai/scale_factor;   

      // until we get more precise/reliable QIE8 simulation...  
      if(totalCharge <= 0.) totalCharge = eps; // protecion against negaive v.
      tshift += HcalTimeSlew::delay(totalCharge, biasSetting);      
      if(tshift <= 0.) tshift = eps;
	  
      if ( cut > -999. ) { //preserve compatibility
	if ((datai > cut) && ( it == 0 || (datai > cs[it-1]))) {
	  // number of TS affected by current move depends on the signal shape:
	  // rising or peaking
	  nts = 2;  // rising
	  if(datai > cs[it+1]) nts = 3; // peaking
	  
	  // 1 or 2 TS to move from here, 
	  // 2d or 3d TS gets leftovers to preserve the sum
	  for (int j = it; j < it+nts && j < maxbin; ++j) {   
 
	    // snippet borrowed from  CaloSamples::offsetTime(offset)
	    // CalibFormats/CaloObjects/src/CaloSamples.cc
	    double t = j*25. - tshift;
	    int firstbin = floor(t/25.);
	    double f = t/25. - firstbin;
	    int nextbin = firstbin + 1;
	    double v1 = (firstbin < 0 || firstbin >= maxbin) ? 0. : cs[firstbin];
	    double v2 = (nextbin  < 0 || nextbin  >= maxbin) ? 0. : cs[nextbin];
	    
	    // Keeping the overal sum intact
	    if(nts == 2) {
	      if(j == it) data[j] = v2*f; 
	      else data[j] =  v1*(1.-f) + v2;
	    }
	    else { // nts = 3
	      if(j == it)       data[j] = v2*f;
	      if(j == it+1)     data[j] = v1*(1.-f) + v2*f;
	      if(j == it+nts-1) data[j] = v1*(1.-f) + v2;
	    }

	  } // end of local move of TS', now update...
	  cs = data;

	} // end of rising edge or peak finding
      }
      else{
	double t=it*25.- tshift;
	int firstbin=floor(t/25.);
	double f=t/25. - firstbin;
	int nextbin = firstbin +1;
	double v2= (nextbin<0 || nextbin >= maxbin) ? 0. : data[nextbin];
	data[it]=v2*f;
	data[it+1]+= (v2-data[it]);
	cs=data;
      }

      if(nts < 3) it++; 
      else it +=2;
    }

    // final update of the shifted TS array 
    cs = data;
  }
}
