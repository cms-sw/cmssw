#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include <iostream>
  
/** Relevant constants are:
  4.5 photoelectrons per MeV (J. Nash's slides)
  APD gain 50, but analog signal stays in GeV
  Account for excess noise factor
 */
 
EcalSimParameterMap::EcalSimParameterMap() :
  theBarrelParameters(2250., 1./2250., 
                   1., 47.6683, 
                   10, 5, true),
  theEndcapParameters( 1800., 1./1800., 
                   1., 47.6683, 
        	       10, 5, true),
  theESParameters(1., 1., 1., 20., 3, 2, false)
{}

EcalSimParameterMap::EcalSimParameterMap(double simHitToPhotoelectronsBarrel, 
                                         double simHitToPhotoelectronsEndcap, 
                                         double photoelectronsToAnalogBarrel, 
                                         double photoelectronsToAnalogEndcap, 
                                         double samplingFactor, double timePhase,
                                         int readoutFrameSize, int binOfMaximum,
                                         bool doPhotostatistics) : 
  theBarrelParameters(simHitToPhotoelectronsBarrel, photoelectronsToAnalogBarrel,
                      samplingFactor, timePhase, 
                      readoutFrameSize, binOfMaximum, doPhotostatistics),
  theEndcapParameters(simHitToPhotoelectronsEndcap, photoelectronsToAnalogEndcap, 
                      samplingFactor, timePhase, 
                      readoutFrameSize, binOfMaximum, doPhotostatistics),
  theESParameters(1., 1., 1., 20., 3, 2, false)
{}

  /*
  CaloSimParameters(double simHitToPhotoelectrons, double photoelectronsToAnalog, 
                 double samplingFactor, double timePhase,
                 int readoutFrameSize, int binOfMaximum,
                 bool doPhotostatistics)
  */
  
const CaloSimParameters & EcalSimParameterMap::simParameters(const DetId & detId) const 
{
  if (EcalSubdetector(detId.subdetId()) == EcalBarrel) 
    return theBarrelParameters;
  else if (EcalSubdetector(detId.subdetId()) == EcalEndcap)
    return theEndcapParameters;
  else 
    return theESParameters;
}

