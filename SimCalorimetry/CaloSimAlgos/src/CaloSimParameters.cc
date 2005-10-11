#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include<iostream>
  
namespace cms {
  std::ostream & operator<<(std::ostream & os, const CaloSimParameters & p) {
    os << "CALO SIM PARAMETERS" << std::endl;
    os << p.simHitToPhotoelectrons() << " SimHit energy per pe" << std::endl;
    os << p.photoelectronsToAnalog() << " Analog signal to be digitized per pe" << std::endl;
    os << "Sampling Factor:      " << p.samplingFactor() << " Incident energy / SimHit Energy " << std::endl;
    return os;
  }
}
