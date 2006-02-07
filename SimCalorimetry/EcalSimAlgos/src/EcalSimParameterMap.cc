#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

  
/** Relevant constants are:
  4.5 photoelectrons per MeV (J. Nash's slides)
  APD gain 50, but analog signal stays in GeV
 */
 
EcalSimParameterMap::EcalSimParameterMap() :
  theBarrelParameters(4500., 1./4500., 
                   1., 47., 
                   10, 5, true),
  theEndcapParameters( 4500., 1./4500., 
                   1., 47., 
                   10, 5, true)
{
}
  /*
  CaloSimParameters(double simHitToPhotoelectrons, double photoelectronsToAnalog, 
                 double samplingFactor, double timePhase,
                 int readoutFrameSize, int binOfMaximum,
                 bool doPhotostatistics)
  */
  

const CaloSimParameters & EcalSimParameterMap::simParameters(const DetId & detId) const {
  return (EcalSubdetector(detId.subdetId()) == EcalBarrel) ? theBarrelParameters : theEndcapParameters;
}
  
