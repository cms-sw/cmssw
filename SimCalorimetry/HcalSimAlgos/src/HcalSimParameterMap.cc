#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
  
namespace cms {

  HcalSimParameterMap::HcalSimParameterMap() :
    theHBHEParameters(0.0005, 1./4.0,
                     117, -2, 
                     10, 5, true),
    theHOParameters( 0.00025, 1./4.0, 217., -2, 10, 5, true),
    theHFParameters1(1., 1./0.3,
                   2.84 , -6,
                  6, 3, false),
    theHFParameters2(1., 1./0.3,
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
  
}
  
