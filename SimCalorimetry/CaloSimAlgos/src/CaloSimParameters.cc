#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include<iostream>
  
std::ostream & operator<<(std::ostream & os, const CaloSimParameters & p) {
  os << "CALO SIM PARAMETERS" << std::endl;
  os << p.simHitToPhotoelectrons() << " pe per SimHit energy " << std::endl;
  os << p.photoelectronsToAnalog() << " Analog signal to be digitized per pe" << std::endl;
  os << "Sampling Factor:      " << p.samplingFactor() << " Incident energy / SimHit Energy " << std::endl;
  return os;
}

