#ifndef LowEnergyFastSimModel_h
#define LowEnergyFastSimModel_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "LowEnergyFastSimParam.h"

#include "G4VFastSimulationModel.hh"
#include "GFlashHitMaker.hh"
#include "G4Region.hh"
#include "G4Types.hh"

class LowEnergyFastSimModel : public G4VFastSimulationModel {
public:
  LowEnergyFastSimModel(const G4String& name, G4Region* region, const edm::ParameterSet& parSet);

  G4bool IsApplicable(const G4ParticleDefinition& particle) override;
  G4bool ModelTrigger(const G4FastTrack& fastTrack) override;
  void DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep) override;

private:
  const G4double fEmax;
  const G4Envelope* const fRegion;
  GFlashHitMaker fHitMaker;
  LowEnergyFastSimParam param;
};

#endif
