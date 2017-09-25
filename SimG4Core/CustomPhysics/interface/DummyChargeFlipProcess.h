
#ifndef DummyChargeFlipProcess_h
#define DummyChargeFlipProcess_h 1
 
#include "globals.hh"
#include "G4HadronicProcess.hh"
#include "G4CrossSectionDataStore.hh"
#include "G4HadronElasticDataSet.hh"
#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4VDiscreteProcess.hh"
#include "G4LightMedia.hh"
#include "G4Step.hh"
#include "G4TrackStatus.hh"

#include <iostream>

class DummyChargeFlipProcess : public G4HadronicProcess
{
public:

   DummyChargeFlipProcess(const G4String& processName = "Dummy");

   ~DummyChargeFlipProcess() override;
 
   G4VParticleChange* PostStepDoIt(const G4Track& aTrack, const G4Step& aStep) override;


   G4bool IsApplicable(const G4ParticleDefinition& aParticleType) override;

   void BuildPhysicsTable(const G4ParticleDefinition& aParticleType) override;

   void DumpPhysicsTable(const G4ParticleDefinition& aParticleType);

private:

   G4double GetMicroscopicCrossSection(const G4DynamicParticle* aParticle,
                                       const G4Element* anElement,
				       G4double aTemp);

};
#endif
