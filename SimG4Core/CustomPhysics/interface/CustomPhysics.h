#ifndef SimG4Core_CustomPhysics_H
#define SimG4Core_CustomPhysics_H
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
#include "G4VPhysicsConstructor.hh"
 
class CustomPhysics : public G4VPhysicsConstructor
{
public:
    CustomPhysics(std::string name,const edm::ParameterSet & p);
    virtual ~CustomPhysics();
protected:
    virtual void ConstructParticle();
    virtual void ConstructProcess();
    void addCustomPhysics();
private:
    edm::ParameterSet m_pCustomPhysics;
};
 
#endif
