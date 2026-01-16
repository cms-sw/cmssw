#ifndef SimG4Core_CustomPhysics_CustomPhysicsListSS_H
#define SimG4Core_CustomPhysics_CustomPhysicsListSS_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "G4VPhysicsConstructor.hh"
#include <string>

class CustomProcessHelper;
class CustomParticleFactory;

class CustomPhysicsListSS : public G4VPhysicsConstructor {
public:
  CustomPhysicsListSS(const std::string& name, const edm::ParameterSet& p, bool useuni = false);
  ~CustomPhysicsListSS() override = default;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  static G4ThreadLocal CustomProcessHelper* myHelper;

  std::unique_ptr<CustomParticleFactory> fParticleFactory;

  bool fHadronicInteraction;

  edm::ParameterSet myConfig;

  std::string particleDefFilePath;
  std::string processDefFilePath;
  double dfactor;
};

#endif
