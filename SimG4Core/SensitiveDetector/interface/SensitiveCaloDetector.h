#ifndef SimG4Core_SensitiveCaloDetector_H
#define SimG4Core_SensitiveCaloDetector_H

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"

#include <vector>
#include <string>

class SensitiveCaloDetector : public SensitiveDetector
{
public:
    SensitiveCaloDetector(std::string & iname, const DDCompactView & cpv,
			  const SensitiveDetectorCatalog & clg,
			  edm::ParameterSet const & p) :
      SensitiveDetector(iname,cpv,clg,p) {}
    virtual void fillHits(edm::PCaloHitContainer &, std::string name = 0) = 0;
};

#endif


