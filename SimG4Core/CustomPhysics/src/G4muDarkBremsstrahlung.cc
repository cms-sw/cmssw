#include "SimG4Core/CustomPhysics/interface/G4muDarkBremsstrahlung.h"
#include "SimG4Core/CustomPhysics/interface/G4muDarkBremsstrahlungModel.h"
#include "SimG4Core/CustomPhysics/interface/G4APrime.h"

//Geant 4
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4LossTableManager.hh"

using namespace std;

G4muDarkBremsstrahlung::G4muDarkBremsstrahlung(const G4String& scalefile,
                                               const G4double biasFactor,
                                               const G4String& name)
    : G4VEmProcess(name), isInitialised(false), mgfile(scalefile), cxBias(biasFactor) {
  G4int subtype = 40;
  SetProcessSubType(subtype);
  SetSecondaryParticle(G4APrime::APrime());
}

G4muDarkBremsstrahlung::~G4muDarkBremsstrahlung() {}

G4bool G4muDarkBremsstrahlung::IsApplicable(const G4ParticleDefinition& p) {
  return (&p == G4MuonPlus::MuonPlus() || &p == G4MuonMinus::MuonMinus());
}

void G4muDarkBremsstrahlung::InitialiseProcess(const G4ParticleDefinition*) {
  if (!isInitialised) {
    AddEmModel(0, new G4muDarkBremsstrahlungModel(mgfile, cxBias));

    isInitialised = true;
    isEnabled = true;
  }
}

void G4muDarkBremsstrahlung::PrintInfo() {}

void G4muDarkBremsstrahlung::SetMethod(std::string method_in) {
  ((G4muDarkBremsstrahlungModel*)EmModel(1))->SetMethod(method_in);
  return;
}

G4bool G4muDarkBremsstrahlung::IsEnabled() { return isEnabled; }

void G4muDarkBremsstrahlung::SetEnable(bool state) {
  isEnabled = state;
  return;
}
