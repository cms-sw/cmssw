
#include "SimG4Core/CustomPhysics/interface/CMSSIMPInelasticProcess.h"
#include "SimG4Core/CustomPhysics/interface/CMSSIMPInelasticXS.h"
#include "SimG4Core/CustomPhysics/interface/CMSSIMP.h"

#include "G4Types.hh"
#include "G4SystemOfUnits.hh"
#include "G4HadProjectile.hh"
#include "G4ElementVector.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4Element.hh"
#include "G4ParticleChange.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleus.hh"

#include "G4HadronicException.hh"
#include "G4HadronicProcessStore.hh"
#include "G4HadronicInteraction.hh"

#include "G4HadronInelasticDataSet.hh"
#include "G4ParticleDefinition.hh"

//////////////////////////////////////////////////////////////////
CMSSIMPInelasticProcess::CMSSIMPInelasticProcess(const G4String& processName)
    : G4HadronicProcess(processName, fHadronic) {
  AddDataSet(new CMSSIMPInelasticXS());
  theParticle = CMSSIMP::SIMP();
}

CMSSIMPInelasticProcess::~CMSSIMPInelasticProcess() {}

G4bool CMSSIMPInelasticProcess::IsApplicable(const G4ParticleDefinition& aP) {
  return theParticle->GetParticleType() == aP.GetParticleType();
}

G4VParticleChange* CMSSIMPInelasticProcess::PostStepDoIt(const G4Track& aTrack, const G4Step&) {
  // if primary is not Alive then do nothing
  theTotalResult->Clear();
  theTotalResult->Initialize(aTrack);
  theTotalResult->ProposeWeight(aTrack.GetWeight());
  if (aTrack.GetTrackStatus() != fAlive) {
    return theTotalResult;
  }

  // Find cross section at end of step and check if <= 0
  //
  G4DynamicParticle* aParticle = const_cast<G4DynamicParticle*>(aTrack.GetDynamicParticle());

  // change this SIMP particle in a neutron
  aParticle->SetPDGcode(2112);
  aParticle->SetDefinition(G4Neutron::Neutron());

  G4Nucleus* target = GetTargetNucleusPointer();
  const G4Material* aMaterial = aTrack.GetMaterial();
  const G4Element* anElement = GetCrossSectionDataStore()->SampleZandA(aParticle, aMaterial, *target);

  // Next check for illegal track status
  //
  if (aTrack.GetTrackStatus() != fAlive && aTrack.GetTrackStatus() != fSuspend) {
    if (aTrack.GetTrackStatus() == fStopAndKill || aTrack.GetTrackStatus() == fKillTrackAndSecondaries ||
        aTrack.GetTrackStatus() == fPostponeToNextEvent) {
      G4ExceptionDescription ed;
      ed << "CMSSIMPInelasticProcess: track in unusable state - " << aTrack.GetTrackStatus() << G4endl;
      ed << "CMSSIMPInelasticProcess: returning unchanged track " << G4endl;
      DumpState(aTrack, "PostStepDoIt", ed);
      G4Exception("CMSSIMPInelasticProcess::PostStepDoIt", "had004", JustWarning, ed);
    }
    // No warning for fStopButAlive which is a legal status here
    return theTotalResult;
  }

  // Initialize the hadronic projectile from the track
  thePro.Initialise(aTrack);
  G4HadronicInteraction* anInteraction = GetHadronicInteractionList()[0];

  G4HadFinalState* result = nullptr;
  G4int reentryCount = 0;

  do {
    try {
      // Call the interaction
      result = anInteraction->ApplyYourself(thePro, *target);
      ++reentryCount;
    } catch (G4HadronicException& aR) {
      G4ExceptionDescription ed;
      aR.Report(ed);
      ed << "Call for " << anInteraction->GetModelName() << G4endl;
      ed << "Target element " << anElement->GetName() << "  Z= " << target->GetZ_asInt()
         << "  A= " << target->GetA_asInt() << G4endl;
      DumpState(aTrack, "ApplyYourself", ed);
      ed << " ApplyYourself failed" << G4endl;
      G4Exception("CMSSIMPInelasticProcess::PostStepDoIt", "had006", FatalException, ed);
    }

    // Check the result for catastrophic energy non-conservation
    result = CheckResult(thePro, *target, result);
    if (reentryCount > 100) {
      G4ExceptionDescription ed;
      ed << "Call for " << anInteraction->GetModelName() << G4endl;
      ed << "Target element " << anElement->GetName() << "  Z= " << target->GetZ_asInt()
         << "  A= " << target->GetA_asInt() << G4endl;
      DumpState(aTrack, "ApplyYourself", ed);
      ed << " ApplyYourself does not completed after 100 attempts" << G4endl;
      G4Exception("CMSSIMPInelasticProcess::PostStepDoIt", "had006", FatalException, ed);
    }
  } while (!result);
  // Check whether kaon0 or anti_kaon0 are present between the secondaries:
  // if this is the case, transform them into either kaon0S or kaon0L,
  // with equal, 50% probability, keeping their dynamical masses (and
  // the other kinematical properties).
  // When this happens - very rarely - a "JustWarning" exception is thrown.
  G4int nSec = result->GetNumberOfSecondaries();
  if (nSec > 0) {
    for (G4int i = 0; i < nSec; ++i) {
      G4DynamicParticle* dynamicParticle = result->GetSecondary(i)->GetParticle();
      const G4ParticleDefinition* particleDefinition = dynamicParticle->GetParticleDefinition();
      if (particleDefinition == G4KaonZero::Definition() || particleDefinition == G4AntiKaonZero::Definition()) {
        G4ParticleDefinition* newPart;
        if (G4UniformRand() > 0.5) {
          newPart = G4KaonZeroShort::Definition();
        } else {
          newPart = G4KaonZeroLong::Definition();
        }
        dynamicParticle->SetDefinition(newPart);
        G4ExceptionDescription ed;
        ed << " Hadronic model " << anInteraction->GetModelName() << G4endl;
        ed << " created " << particleDefinition->GetParticleName() << G4endl;
        ed << " -> forced to be " << newPart->GetParticleName() << G4endl;
        G4Exception("G4HadronicProcess::PostStepDoIt", "had007", JustWarning, ed);
      }
    }
  }

  result->SetTrafoToLab(thePro.GetTrafoToLab());

  ClearNumberOfInteractionLengthLeft();

  FillResult(result, aTrack);

  return theTotalResult;
}
