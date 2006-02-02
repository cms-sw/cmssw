#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

  
/** Relevant constants are:
  4.5 photoelectrons per MeV (J. Nash's slides)
  APD gain 50, but analog signal stays in GeV
 */
 
EcalSimParameterMap::EcalSimParameterMap() :
  theBarrelParameters(2250., 1./2250., 
                   1., 47., 
                   10, 5, true),
  theEndcapParameters( 1800., 1./1800., 
                   1., 47., 
                   10, 5, true)
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
  
