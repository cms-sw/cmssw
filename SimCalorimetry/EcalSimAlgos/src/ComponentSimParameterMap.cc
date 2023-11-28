#include "SimCalorimetry/EcalSimAlgos/interface/ComponentSimParameterMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include <iostream>

ComponentSimParameterMap::ComponentSimParameterMap(bool addToBarrel,
                                                   bool separateDigi,
                                                   double simHitToPhotoelectronsBarrel,
                                                   double simHitToPhotoelectronsEndcap,
                                                   double photoelectronsToAnalogBarrel,
                                                   double photoelectronsToAnalogEndcap,
                                                   double samplingFactor,
                                                   double timePhase,
                                                   int readoutFrameSize,
                                                   int binOfMaximum,
                                                   bool doPhotostatistics,
                                                   bool syncPhase)
    : m_addToBarrel(addToBarrel),
      m_separateDigi(separateDigi),
      theComponentParameters(simHitToPhotoelectronsBarrel,
                             photoelectronsToAnalogBarrel,
                             samplingFactor,
                             timePhase,
                             readoutFrameSize,
                             binOfMaximum,
                             doPhotostatistics,
                             syncPhase) {}

const CaloSimParameters& ComponentSimParameterMap::simParameters(const DetId& detId) const {
  return theComponentParameters;
}
