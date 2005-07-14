#ifndef SimG4Core_PhysicsList_H
#define SimG4Core_PhysicsList_H

#include "G4VModularPhysicsList.hh"

class PhysicsList : public G4VModularPhysicsList
{
public:
    PhysicsList();
    virtual ~PhysicsList();
    virtual void SetCuts();
};

#endif
