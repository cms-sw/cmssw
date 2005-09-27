#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

namespace cms {
  
  EcalSimParameterMap::EcalSimParameterMap() :
    theBarrelParameters(1/2250., 2250., 
                     1., 0., 
                     10, 6, true),
    theEndcapParameters( 1/1800., 1800., 
                     1., 0., 
                     10, 6, true)
  {
  }
  /*
    CaloSimParameters(double photomultiplierGain, double amplifierGain,
                   double samplingFactor, double peakTime,
                   int readoutFrameSize, int binOfMaximum,
                   bool doPhotostatistics)
  */
  
  
  const CaloSimParameters & EcalSimParameterMap::simParameters(const DetId & detId) const {
    return (EcalSubdetector(detId.subdetId()) == EcalBarrel) ? theBarrelParameters : theEndcapParameters;
  }
  
} 
  
