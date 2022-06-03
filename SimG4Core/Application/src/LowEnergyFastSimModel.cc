#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Application/interface/LowEnergyFastSimModel.h"
#include "SimG4Core/Application/interface/TrackingAction.h"

#include "G4VFastSimulationModel.hh"
#include "G4EventManager.hh"
#include "G4Electron.hh"
#include "GFlashHitMaker.hh"
#include "G4Region.hh"
#include "G4Material.hh"
#include "G4Positron.hh"
#include "G4ParticleDefinition.hh"
#include "G4PhysicalConstants.hh"

constexpr G4double twomass = 2 * CLHEP::electron_mass_c2;

LowEnergyFastSimModel::LowEnergyFastSimModel(const G4String& name, G4Region* region, const edm::ParameterSet& parSet)
    : G4VFastSimulationModel(name, region),
      fRegion(region),
      fTrackingAction(nullptr),
      fCheck(false),
      fTailPos(0., 0., 0.) {
  fEmax = parSet.getParameter<double>("LowEnergyGflashEcalEmax") * CLHEP::GeV;
  fPositron = G4Positron::Positron();
  fMaterial = nullptr;
  auto table = G4Material::GetMaterialTable();
  for (auto& mat : *table) {
    G4String nam = mat->GetName();
    size_t n = nam.size();
    if (n > 4) {
      G4String sn = nam.substr(n - 5, 5);
      if (sn == "PbWO4") {
        fMaterial = mat;
        break;
      }
    }
  }
  G4String nm = (nullptr == fMaterial) ? "not found!" : fMaterial->GetName();
  edm::LogVerbatim("LowEnergyFastSimModel") << "LowEGFlash material: <" << nm << ">";
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
  /*
  edm::LogVerbatim("LowEnergyFastSimModel") << track->GetDefinition()->GetParticleName()
					    << " Ekin(MeV)=" << energy << " material: <"
                                            << track->GetMaterial()->GetName() << ">";
  */
  return (energy < fEmax && fMaterial == track->GetMaterial());
}

void LowEnergyFastSimModel::DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep) {
  fastStep.KillPrimaryTrack();
  fastStep.SetPrimaryTrackPathLength(0.0);
  auto track = fastTrack.GetPrimaryTrack();
  G4double energy = track->GetKineticEnergy();

  const G4ThreeVector& pos = track->GetPosition();

  G4double inPointEnergy = fParam.GetInPointEnergyFraction(energy) * energy;

  // take into account positron annihilation (not included in in-point)
  if (fPositron == track->GetDefinition())
    energy += twomass;

  const G4ThreeVector& momDir = track->GetMomentumDirection();

  // in point energy deposition
  GFlashEnergySpot spot;
  spot.SetEnergy(inPointEnergy);
  spot.SetPosition(pos);
  fHitMaker.make(&spot, &fastTrack);

  // tail energy deposition
  const G4double etail = energy - inPointEnergy;
  const G4int nspots = etail;
  const G4double tailEnergy = etail / (nspots + 1);
  /*  
  edm::LogVerbatim("LowEnergyFastSimModel") << track->GetDefinition()->GetParticleName()
					    << " Ekin(MeV)=" << energy << " material: <"
                                            << track->GetMaterial()->GetName() 
					    << "> Elocal=" << inPointEnergy
					    << " Etail=" << tailEnergy
					    << " Nspots=" << nspots+1;
  */
  for (G4int i = 0; i <= nspots; ++i) {
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
