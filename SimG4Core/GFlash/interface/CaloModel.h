#ifndef SimG4Core_GFlash_CaloModel_H
#define SimG4Core_GFlash_CaloModel_H

// Joanna Weng 08.2005
// setup of volumes for GFLASH

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
 
#ifdef G4V7
class GFlashHomoShowerParamterisation;
#else
class GFlashHomoShowerParameterisation;
#endif
class GFlashHitMaker;
class GFlashParticleBounds;
class GflashEMShowerModel;
class GflashHadronShowerModel;

/* observes the world volume (standard COBRA  Observer) and creates  
theFastShowerModel and theParametrisation for the logical volumes 
named in OscarApplication/G4SimApplication/test/ShowerModelVolumes.xml  */
class CaloModel 
{
public:
  CaloModel(G4LogicalVolumeToDDLogicalPartMap& ,edm::ParameterSet const &);
  ~CaloModel();
private:
  void build();  
#ifdef G4V7
  GFlashHomoShowerParamterisation *theParametrisation;
#else
  GFlashHomoShowerParameterisation *theParameterisation;
#endif
  edm::ParameterSet m_pCaloModel;
  G4LogicalVolumeToDDLogicalPartMap map_;
  GFlashHitMaker *theHMaker;
  GFlashParticleBounds *theParticleBounds;
  GflashEMShowerModel *theShowerModel;
  GflashHadronShowerModel *theHadronShowerModel;
};

#endif
