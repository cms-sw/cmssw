#ifndef SimG4Core_CustomPhysicsList_H
#define SimG4Core_CustomPhysicsList_H
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
#include "G4VPhysicsConstructor.hh"
 
class CustomPhysicsList : public G4VPhysicsConstructor
{
public:
    CustomPhysicsList(std::string name,const edm::ParameterSet & p);
    virtual ~CustomPhysicsList();
protected:
    virtual void ConstructParticle();
    virtual void ConstructProcess();
    void addCustomPhysicsList();
private:
    edm::ParameterSet m_pCustomPhysicsList;
};
 
#endif
