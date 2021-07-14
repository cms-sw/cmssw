#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Application/interface/LowEnergyFastSimModel.h"
#include "SimG4Core/Application/interface/TrackingAction.h"

#include "G4VFastSimulationModel.hh"
#include "G4EventManager.hh"
#include "G4Electron.hh"
#include "GFlashHitMaker.hh"
#include "G4Region.hh"
#include "G4PhysicalConstants.hh"

constexpr G4double twomass = 2 * CLHEP::electron_mass_c2;

LowEnergyFastSimModel::LowEnergyFastSimModel(const G4String& name, G4Region* region, const edm::ParameterSet& parSet)
    : G4VFastSimulationModel(name, region),
      fRegion(region),
      fTrackingAction(nullptr),
      fCheck(false),
      fTailPos(0., 0., 0.) {
  fEmax = parSet.getParameter<double>("LowEnergyGflashEcalEmax") * CLHEP::GeV;
}

G4bool LowEnergyFastSimModel::IsApplicable(const G4ParticleDefinition& particle) {
  return (11 == std::abs(particle.GetPDGEncoding()));
}

G4bool LowEnergyFastSimModel::ModelTrigger(const G4FastTrack& fastTrack) {
  const G4Track* track = fastTrack.GetPrimaryTrack();
  if (fCheck) {
    if (nullptr == fTrackingAction) {
      fTrackingAction = static_cast<const TrackingAction*>(G4EventManager::GetEventManager()->GetUserTrackingAction());
    }
    int pdgMother = std::abs(fTrackingAction->geant4Track()->GetDefinition()->GetPDGEncoding());
    if (pdgMother == 11 || pdgMother == 22)
      return false;
  }
  G4double energy = track->GetKineticEnergy();
  return (energy < fEmax && fRegion == fastTrack.GetEnvelope());
}

void LowEnergyFastSimModel::DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep) {
  fastStep.KillPrimaryTrack();
  fastStep.SetPrimaryTrackPathLength(0.0);
  G4double energy = fastTrack.GetPrimaryTrack()->GetKineticEnergy();

  const G4ThreeVector& pos = fastTrack.GetPrimaryTrack()->GetPosition();

  G4double inPointEnergy = fParam.GetInPointEnergyFraction(energy) * energy;

  // take into account positron annihilation (not included in in-point)
  if (-11 == fastTrack.GetPrimaryTrack()->GetDefinition()->GetPDGEncoding())
    energy += twomass;

  const G4ThreeVector& momDir = fastTrack.GetPrimaryTrack()->GetMomentumDirection();

  // in point energy deposition
  GFlashEnergySpot spot;
  spot.SetEnergy(inPointEnergy);
  spot.SetPosition(pos);
  fHitMaker.make(&spot, &fastTrack);

  // tail energy deposition
  G4double etail = energy - inPointEnergy;
  const G4int nspots = (G4int)(etail) + 1;
  const G4double tailEnergy = etail / (G4double)nspots;
  for (G4int i = 0; i < nspots; ++i) {
    const G4double r = fParam.GetRadius(energy);
    const G4double z = fParam.GetZ();

    const G4double phi = CLHEP::twopi * G4UniformRand();
    fTailPos.set(r * std::cos(phi), r * std::sin(phi), z);
    fTailPos.rotateUz(momDir);
    fTailPos += pos;

    spot.SetEnergy(tailEnergy);
    spot.SetPosition(fTailPos);
    fHitMaker.make(&spot, &fastTrack);
  }
}
