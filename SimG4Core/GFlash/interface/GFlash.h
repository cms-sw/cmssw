#ifndef SimG4Core_GFlash_GFlash_H
#define SimG4Core_GFlash_GFlash_H
// Joanna Weng 08.2005
// setup of volumes for GFLASH

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class CaloModel;

class GFlash : public PhysicsList
{
public:
  GFlash(G4LogicalVolumeToDDLogicalPartMap& map, const edm::ParameterSet & p);
  virtual ~GFlash();
private:
  CaloModel * caloModel;   
};

#endif

