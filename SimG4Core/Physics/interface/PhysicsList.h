#ifndef SimG4Core_PhysicsList_H
#define SimG4Core_PhysicsList_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4VModularPhysicsList.hh"

class DDG4ProductionCuts;

class PhysicsList : public G4VModularPhysicsList
{
public:
    PhysicsList(const edm::ParameterSet & p);
    virtual ~PhysicsList();
    virtual void SetCuts();
private:
    edm::ParameterSet m_pPhysics; 
    DDG4ProductionCuts * prodCuts;
};

#endif
