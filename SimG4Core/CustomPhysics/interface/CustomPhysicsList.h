#ifndef SimG4Core_CustomPhysicsList_H
#define SimG4Core_CustomPhysicsList_H

#include "SimG4Core/CustomPhysics/interface/HadronicProcessHelper.hh"

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
#include "G4VPhysicsConstructor.hh"

class G4ProcessHelper;
class G4Decay;

class CustomPhysicsList : public G4VPhysicsConstructor 
{
public:
  CustomPhysicsList(std::string name, const edm::ParameterSet & p);
  virtual ~CustomPhysicsList();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:

  static G4ThreadLocal G4Decay* fDecayProcess;
  static G4ThreadLocal G4ProcessHelper* myHelper;
  static G4ThreadLocal bool fInitialized;

  bool fHadronicInteraction;

  edm::ParameterSet myConfig;

  std::string particleDefFilePath;
  std::string processDefFilePath;
};
 
#endif
