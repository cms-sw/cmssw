#ifndef SimG4Core_GFlash_GFlash_H
#define SimG4Core_GFlash_GFlash_H
// Joanna Weng 08.2005
// setup of volumes for GFLASH

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class GFlash : public PhysicsList
{

public:
    GFlash(const edm::ParameterSet & p);
    virtual ~GFlash();
};

#endif

