#ifndef SimG4Core_CustomPhysicsListSS_H
#define SimG4Core_CustomPhysicsListSS_H

#include "SimG4Core/CustomPhysics/interface/HadronicProcessHelper.hh"

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
#include "G4VPhysicsConstructor.hh"

class G4ProcessHelper;
class G4Decay;

class CustomPhysicsListSS : public G4VPhysicsConstructor 
{
public:
  CustomPhysicsListSS(std::string name, const edm::ParameterSet & p);
  ~CustomPhysicsListSS() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:

  static G4ThreadLocal G4Decay* fDecayProcess;
  static G4ThreadLocal G4ProcessHelper* myHelper;

  bool fHadronicInteraction;

  edm::ParameterSet myConfig;

  std::string particleDefFilePath;
  std::string processDefFilePath;

};
 
#endif
