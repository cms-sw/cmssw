#ifndef SimG4Core_CustomPhysics_CustomPhysicsListSS_H
#define SimG4Core_CustomPhysics_CustomPhysicsListSS_H

#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "G4VPhysicsConstructor.hh"
#include <string>

class G4ProcessHelper;
class G4Decay;
class CustomParticleFactory;

class CustomPhysicsListSS : public G4VPhysicsConstructor 
{
public:
  CustomPhysicsListSS(const std::string& name, const edm::ParameterSet & p,
		      bool useuni = false);
  ~CustomPhysicsListSS() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:

  static G4ThreadLocal std::unique_ptr<G4Decay> fDecayProcess;
  static G4ThreadLocal std::unique_ptr<G4ProcessHelper> myHelper;

  std::unique_ptr<CustomParticleFactory> fParticleFactory;

  bool fHadronicInteraction;

  edm::ParameterSet myConfig;

  std::string particleDefFilePath;
  std::string processDefFilePath;
  double dfactor;
};
 
#endif
