#ifndef SimG4Core_DummyChargeFlipProcess_H
#define SimG4Core_DummyChargeFlipProcess_H
 
#include "G4HadronicProcess.hh"
#include "G4CrossSectionDataStore.hh"
#include "G4HadronElasticDataSet.hh"
#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4VDiscreteProcess.hh"
#include "G4LightMedia.hh"
#include "G4Step.hh"
#include "G4TrackStatus.hh"

class DummyChargeFlipProcess : public G4HadronicProcess
{
public:
    DummyChargeFlipProcess(const std::string& processName = "LElastic");
    ~DummyChargeFlipProcess(); 
    G4VParticleChange * PostStepDoIt(const G4Track & aTrack, const G4Step & aStep);
    bool IsApplicable(const G4ParticleDefinition & aParticleType);
    void BuildPhysicsTable(const G4ParticleDefinition & aParticleType);
    void DumpPhysicsTable(const G4ParticleDefinition & aParticleType);
private:
    double GetMicroscopicCrossSection(const G4DynamicParticle * aParticle,
				      const G4Element * anElement,
				      double aTemp);

};

#endif
