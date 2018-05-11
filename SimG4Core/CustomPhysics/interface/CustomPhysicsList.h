#ifndef SimG4Core_CustomPhysics_CustomPhysicsList_H
#define SimG4Core_CustomPhysics_CustomPhysicsList_H

#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "G4VPhysicsConstructor.hh"

#include <string>

class G4ProcessHelper;
class CustomParticleFactory;

class CustomPhysicsList : public G4VPhysicsConstructor 
{
public:
  CustomPhysicsList(const std::string& name, const edm::ParameterSet & p, 
		    bool useuni = false);
  ~CustomPhysicsList() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:

  static G4ThreadLocal std::unique_ptr<G4ProcessHelper> myHelper;
  std::unique_ptr<CustomParticleFactory> fParticleFactory;

  bool fHadronicInteraction;

  edm::ParameterSet myConfig;

  std::string particleDefFilePath;
  std::string processDefFilePath;
  double dfactor;
};
 
#endif
