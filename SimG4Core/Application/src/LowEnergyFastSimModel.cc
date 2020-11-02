#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Application/interface/LowEnergyFastSimModel.h"

#include "G4VFastSimulationModel.hh"
#include "G4Electron.hh"
#include "GFlashHitMaker.hh"
#include "G4Region.hh"
#include "G4PhysicalConstants.hh"

constexpr double twomass = 2 * CLHEP::electron_mass_c2;

LowEnergyFastSimModel::LowEnergyFastSimModel(const G4String& name, G4Region* region, const edm::ParameterSet& parSet)
    : G4VFastSimulationModel(name, region),
      fEmax(parSet.getParameter<double>("LowEnergyGflashEcalEmax")),
      fRegion(region) {}

G4bool LowEnergyFastSimModel::IsApplicable(const G4ParticleDefinition& particle) {
  return (11 == std::abs(particle.GetPDGEncoding()));
}

G4bool LowEnergyFastSimModel::ModelTrigger(const G4FastTrack& fastTrack) {
  G4double energy = fastTrack.GetPrimaryTrack()->GetKineticEnergy();
  return energy < fEmax && fRegion == fastTrack.GetEnvelope();
}

void LowEnergyFastSimModel::DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep) {
  fastStep.KillPrimaryTrack();
  fastStep.SetPrimaryTrackPathLength(0.0);
  double energy = fastTrack.GetPrimaryTrack()->GetKineticEnergy();

  const G4ThreeVector& pos = fastTrack.GetPrimaryTrack()->GetPosition();

  G4double inPointEnergy = param.GetInPointEnergyFraction(energy) * energy;

  // take into account positron annihilation (not included in in-point)
  if (-11 == fastTrack.GetPrimaryTrack()->GetDefinition()->GetPDGEncoding())
    energy += twomass;

  const G4ThreeVector& momDir = fastTrack.GetPrimaryTrack()->GetMomentumDirection();
  const G4ThreeVector& ortho = momDir.orthogonal();
  const G4ThreeVector& cross = momDir.cross(ortho);

  // in point energy deposition
  GFlashEnergySpot spot;
  spot.SetEnergy(inPointEnergy);
  spot.SetPosition(pos);
  fHitMaker.make(&spot, &fastTrack);

  // tail energy deposition
  G4double etail = energy - inPointEnergy;
  const G4int nspots = int(etail) + 1;
  const G4double tailEnergy = etail / (G4double)nspots;
  for (G4int i = 0; i < nspots; ++i) {
    const G4double radius = param.GetRadius(energy);
    const G4double z = param.GetZ();

    const G4double phi = CLHEP::twopi * G4UniformRand();
    const G4ThreeVector tailPos = pos + z * momDir + radius * std::cos(phi) * ortho + radius * std::sin(phi) * cross;

    spot.SetEnergy(tailEnergy);
    spot.SetPosition(tailPos);
    fHitMaker.make(&spot, &fastTrack);
  }
}
