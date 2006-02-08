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
	          10, 5, true),
  theESParameters(1., 1., 1., 20., 3, 2, false)
{
}
  /*
  CaloSimParameters(double simHitToPhotoelectrons, double photoelectronsToAnalog, 
                 double samplingFactor, double timePhase,
                 int readoutFrameSize, int binOfMaximum,
                 bool doPhotostatistics)
  */
  

const CaloSimParameters & EcalSimParameterMap::simParameters(const DetId & detId) const {
  if (EcalSubdetector(detId.subdetId()) == EcalBarrel) 
    return theBarrelParameters;
  else if (EcalSubdetector(detId.subdetId()) == EcalEndcap)
    return theEndcapParameters;
  else 
    return theESParameters;
  // return (EcalSubdetector(detId.subdetId()) == EcalBarrel) ? theBarrelParameters : theEndcapParameters;
}
  
