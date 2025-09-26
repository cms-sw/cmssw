#include "SimG4Core/CustomPhysics/interface/CMSmuDarkBremsstrahlung.h"
#include "SimG4Core/CustomPhysics/interface/CMSmuDarkBremsstrahlungModel.h"
#include "SimG4Core/CustomPhysics/interface/CMSAPrime.h"

//Geant 4
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4LossTableManager.hh"

using namespace std;

CMSmuDarkBremsstrahlung::CMSmuDarkBremsstrahlung(const G4String& scalefile,
                                                 const G4double biasFactor,
                                                 const G4String& name)
    : G4VEmProcess(name), isInitialised(false), mgfile(scalefile), cxBias(biasFactor) {
  G4int subtype = 40;
  SetProcessSubType(subtype);
  SetSecondaryParticle(CMSAPrime::APrime());
}

G4bool CMSmuDarkBremsstrahlung::IsApplicable(const G4ParticleDefinition& p) {
  return (&p == G4MuonPlus::MuonPlus() || &p == G4MuonMinus::MuonMinus());
}

void CMSmuDarkBremsstrahlung::InitialiseProcess(const G4ParticleDefinition*) {
  if (!isInitialised) {
    AddEmModel(0, new CMSmuDarkBremsstrahlungModel(mgfile, cxBias));

    isInitialised = true;
    isEnabled = true;
  }
}

void CMSmuDarkBremsstrahlung::SetMethod(std::string method_in) {
  ((CMSmuDarkBremsstrahlungModel*)EmModel(1))->SetMethod(method_in);
  return;
}

G4bool CMSmuDarkBremsstrahlung::IsEnabled() { return isEnabled; }

void CMSmuDarkBremsstrahlung::SetEnable(bool state) {
  isEnabled = state;
  return;
}
