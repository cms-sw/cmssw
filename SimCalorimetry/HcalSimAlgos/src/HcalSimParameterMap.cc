#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
  

// the second value, amplifierGain, maybe should be 1/(4 pe/fC) in barrel,
// and 1/(0.3 pe/fC) in the endcap.  I'll change them to make them
// consistent with the current default calibrations.
HcalSimParameterMap::HcalSimParameterMap() :
  theHBHEParameters(2000., 0.3305,
                   117, -2, 
                   10, 5, true),
  theHOParameters( 4000., 0.3065, 217., -2, 10, 5, true),
  theHFParameters1(1., 18.93,
                 2.84 , -6,
                6, 3, false),
  theHFParameters2(1., 13.93,
                 2.09 , -6,
                6, 3, false)
{
}
/*
  CaloSimParameters(double photomultiplierGain, double amplifierGain,
                 double samplingFactor, double timePhase,
                 int readoutFrameSize, int binOfMaximum,
                 bool doPhotostatistics)
*/


const CaloSimParameters & HcalSimParameterMap::simParameters(const DetId & detId) const {
  HcalDetId hcalDetId(detId);
  if(hcalDetId.subdet() == HcalBarrel || hcalDetId.subdet() == HcalEndcap) {
     return theHBHEParameters;
  } else if(hcalDetId.subdet() == HcalOuter) {
     return theHOParameters;
  } else { // HF
    if(hcalDetId.depth() == 1) {
      return theHFParameters1;
    } else {
      return theHFParameters2;
    }
  }
}

