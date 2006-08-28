#ifndef SimG4Core_GFlash_CaloModel_H
#define SimG4Core_GFlash_CaloModel_H

// Joanna Weng 08.2005
// setup of volumes for GFLASH

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Geometry/interface/DDDWorld.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class GFlashHomoShowerParamterisation;
class GFlashHitMaker;
class GFlashShowerModel;
class GFlashParticleBounds;

/* observes the world volume (standard COBRA  Observer) and creates  
theFastShowerModel and theParametrisation for the logical volumes 
named in OscarApplication/G4SimApplication/test/ShowerModelVolumes.xml  */
class CaloModel : public Observer<const DDDWorld *>
{
public:
    CaloModel(edm::ParameterSet const & p);
    ~CaloModel();
    void update(const DDDWorld * w);  
private:
    GFlashHomoShowerParamterisation *theParametrisation;
    GFlashHitMaker *theHMaker;
    GFlashParticleBounds *theParticleBounds;
    GFlashShowerModel *theShowerModel;  	
    edm::ParameterSet m_pCaloModel;
};

#endif
