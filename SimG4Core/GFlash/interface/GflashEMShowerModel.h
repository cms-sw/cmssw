//
//---------------------------------------------------------------
//  GEANT 4 class header file
//
//  GflashEMShowerModel
//
//  Class description:
//
//  GFlash parameterisation shower model.

// Authors: E.Barberio & Joanna Weng - 9.11.04
// other authors : Soon Yung Jun & Dongwook Jang - 2007/12/07
//---------------------------------------------------------------

#ifndef GflashEMShowerModel_h
#define GflashEMShowerModel_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "G4VFastSimulationModel.hh"

#include "G4Navigator.hh"
#include "G4Step.hh"
#include "G4TouchableHandle.hh"

class GflashEMShowerProfile;
class G4Region;

class GflashEMShowerModel : public G4VFastSimulationModel {
public:
  GflashEMShowerModel(const G4String &name, G4Envelope *env, const edm::ParameterSet &parSet);
  ~GflashEMShowerModel() override;

  G4bool ModelTrigger(const G4FastTrack &) override;
  G4bool IsApplicable(const G4ParticleDefinition &) override;
  void DoIt(const G4FastTrack &, G4FastStep &) override;

private:
  G4bool excludeDetectorRegion(const G4FastTrack &fastTrack);
  void makeHits(const G4FastTrack &fastTrack);
  void updateGflashStep(const G4ThreeVector &position, G4double time);

private:
  edm::ParameterSet theParSet;

  GflashEMShowerProfile *theProfile;

  const G4Region *theRegion;

  G4Step *theGflashStep;
  G4Navigator *theGflashNavigator;
  G4TouchableHandle theGflashTouchableHandle;
};
#endif
