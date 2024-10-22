#include "SimG4Core/Notification/interface/GenParticleInfoExtractor.h"
#include "G4PrimaryParticle.hh"

const GenParticleInfo &GenParticleInfoExtractor::operator()(const G4PrimaryParticle *p) const {
  G4VUserPrimaryParticleInformation *up = p->GetUserInformation();
  GenParticleInfo *gpi = dynamic_cast<GenParticleInfo *>(up);
  if (up == nullptr) {
    G4Exception("SimG4Core/Notification",
                "mc001",
                FatalException,
                "GenParticleInfoExtractor: G4PrimaryParticle has no user information");
  } else if (gpi == nullptr) {
    G4Exception("SimG4Core/Notification",
                "mc001",
                FatalException,
                "GenParticleInfoExtractor: user information in G4PrimaryParticle is not of GenParticleInfo type");
  }
  // Silence Clang analyzer warning: G4Exception will be thrown if gpi is null
  [[clang::suppress]] return *gpi;
}
