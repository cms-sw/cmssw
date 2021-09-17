#include "SimG4Core/PhysicsLists/interface/CMSTrackingCutModel.h"

#include "G4ParticleDefinition.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"

CMSTrackingCutModel::CMSTrackingCutModel(const G4ParticleDefinition* part)
    : particle_(part), deltaE_(0.0), factor_(0.0), rms_(0.0) {
  if (part == G4Positron::Positron()) {
    deltaE_ = 2 * CLHEP::electron_mass_c2;
  }
}

CMSTrackingCutModel::~CMSTrackingCutModel() {}

G4double CMSTrackingCutModel::SampleEnergyDepositEcal(G4double kinEnergy) {
  G4double edep = kinEnergy * factor_;
  if (rms_ > 0.) {
    edep *= G4RandGauss::shoot(1.0, rms_);
  }
  return edep + deltaE_;
}
