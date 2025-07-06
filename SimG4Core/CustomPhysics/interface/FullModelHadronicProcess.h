#ifndef SimG4Core_CustomPhysics_FullModelHadronicProcess_h
#define SimG4Core_CustomPhysics_FullModelHadronicProcess_h 1

#include "globals.hh"
#include "G4VDiscreteProcess.hh"
#include "G4EnergyRangeManager.hh"
#include "G4Nucleus.hh"
#include "G4ReactionProduct.hh"
#include <vector>

#include "SimG4Core/CustomPhysics/interface/FullModelReactionDynamics.h"

class CustomProcessHelper;

class FullModelHadronicProcess : public G4VDiscreteProcess {
public:
  FullModelHadronicProcess(CustomProcessHelper *aHelper, const G4String &processName = "FullModelHadronicProcess");

  ~FullModelHadronicProcess() override = default;

  G4bool IsApplicable(const G4ParticleDefinition &aP) override;

  G4VParticleChange *PostStepDoIt(const G4Track &aTrack, const G4Step &aStep) override;

protected:
  G4double GetMeanFreePath(const G4Track &aTrack, G4double, G4ForceCondition *) override;

private:
  void CalculateMomenta(G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> &vec,
                        G4int &vecLen,
                        const G4HadProjectile *originalIncident,
                        const G4DynamicParticle *originalTarget,
                        G4ReactionProduct &modifiedOriginal,
                        G4Nucleus &targetNucleus,
                        G4ReactionProduct &currentParticle,
                        G4ReactionProduct &targetParticle,
                        G4bool &incidentHasChanged,
                        G4bool &targetHasChanged,
                        G4bool quasiElastic);

  G4bool MarkLeadingStrangeParticle(const G4ReactionProduct &currentParticle,
                                    const G4ReactionProduct &targetParticle,
                                    G4ReactionProduct &leadParticle);

  void Rotate(G4FastVector<G4ReactionProduct, MYGHADLISTSIZE> &vec, G4int &vecLen);

  CustomProcessHelper *theHelper;
  const G4ParticleDefinition *theParticle{nullptr};
  G4ParticleDefinition *newParticle{nullptr};

  G4Nucleus targetNucleus;
  FullModelReactionDynamics theReactionDynamics;

  G4ThreeVector incomingCloud3Momentum;
  std::vector<G4double> xsec_;
};

#endif
