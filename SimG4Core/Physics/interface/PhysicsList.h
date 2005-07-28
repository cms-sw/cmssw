#ifndef SimG4Core_PhysicsList_H
#define SimG4Core_PhysicsList_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4VModularPhysicsList.hh"

#include "SealKernel/Component.h"
 
class PhysicsList : public G4VModularPhysicsList, public seal::Component
{
    DECLARE_SEAL_COMPONENT;
public:
    PhysicsList(seal::Context * c, const edm::ParameterSet & p);
    virtual ~PhysicsList();
    virtual void SetCuts();
private:
    seal::Context * m_context;
    edm::ParameterSet m_pPhysics; 
};

#endif
