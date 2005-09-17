#include "SimCalorimetry/HcalSimAlgos/interface/HcalQIESim.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"

namespace cms {
  
  HcalQIESim::HcalQIESim() 
  : theStartingCapId(0)
  {
  }
  
  
  HcalQIESample HcalQIESim::makeSample(const DetId & id, int tbin, double signal) {
    int capId = (theStartingCapId + tbin) % 4;
  
  /*
    HcalConversionService * converter = HcalConversionService::instance();
    double pedestalRMS = converter->pedestalRMS(cell, capId);
    double gainRMS = converter->gainRMS(cell, capId);
  
    double pedestalError = theRandomGaussian.fire(0., pedestalRMS);
    double gainError = theRandomGaussian.fire(0., gainRMS);
  
    double jitteredSignal = (signal+pedestalError) * gainError;
    int adc = converter->FC2ADC(cell, capId, signal);
   */ 
  
    int pedestal = 6;
    // get it from the SimParameterMap
    double gain = 1.;
    int adc = (int)(pedestal + gain*signal);
    return HcalQIESample(adc, capId, 0, 0);
  }
    
}

