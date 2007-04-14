#ifndef SimG4Core_CustomPhysicsList_H
#define SimG4Core_CustomPhysicsList_H

#include "SimG4Core/CustomPhysics/interface/HadronicProcessHelper.hh"
#include "SimG4Core/CustomPhysics/interface/G4ProcessHelper.hh"

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
#include "G4VPhysicsConstructor.hh"

class CustomPhysicsList : public G4VPhysicsConstructor 
{
public:
  CustomPhysicsList(std::string name, const edm::ParameterSet & p);
  virtual ~CustomPhysicsList();
protected:
    virtual void ConstructParticle();
    virtual void ConstructProcess();
    void addCustomPhysics();
//    void SetCuts();

private:

    void setupRHadronPhycis(G4ParticleDefinition* particle);
    void setupSUSYPhycis(G4ParticleDefinition* particle);

    //HadronicProcessHelper *myHelper;
    G4ProcessHelper *myHelper;

   edm::ParameterSet myConfig;

   std::string particleDefFilePath;
   std::string processDefFilePath;

};
 
#endif
