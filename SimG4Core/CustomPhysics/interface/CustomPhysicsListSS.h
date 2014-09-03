#ifndef SimG4Core_CustomPhysicsListSS_H
#define SimG4Core_CustomPhysicsListSS_H

#include "SimG4Core/CustomPhysics/interface/HadronicProcessHelper.hh"

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
#include "G4VPhysicsConstructor.hh"

class G4ProcessHelper;

class CustomPhysicsListSS : public G4VPhysicsConstructor 
{
public:
  CustomPhysicsListSS(std::string name, const edm::ParameterSet & p);
  virtual ~CustomPhysicsListSS();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

protected:

  void addCustomPhysics();

private:

  void setupRHadronPhycis(G4ParticleDefinition* particle);
  void setupSUSYPhycis(G4ParticleDefinition* particle);

  G4ProcessHelper *myHelper;

  edm::ParameterSet myConfig;

  std::string particleDefFilePath;
  std::string processDefFilePath;

};
 
#endif
