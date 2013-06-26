#ifndef SimG4Core_SensitiveTkDetector_H
#define SimG4Core_SensitiveTkDetector_H

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"

#include <vector>
#include <string>

class SensitiveTkDetector : public SensitiveDetector
{
public:
    SensitiveTkDetector(std::string & iname, const DDCompactView & cpv,
			SensitiveDetectorCatalog & clg, 
			edm::ParameterSet const & p) : 
      SensitiveDetector(iname, cpv, clg, p) {}
    virtual void fillHits(edm::PSimHitContainer &, std::string name = 0) = 0;
};

#endif




