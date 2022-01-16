#include "SimG4Core/Notification/interface/GenParticleInfoExtractor.h"
#include "G4PrimaryParticle.hh"

const GenParticleInfo &GenParticleInfoExtractor::operator()(const G4PrimaryParticle *p) const {
  G4VUserPrimaryParticleInformation *up = p->GetUserInformation();
  if (up == nullptr)
    G4Exception("SimG4Core/Notification", "mc001", FatalException, 
                "GenParticleInfoExtractor: G4PrimaryParticle has no user information");
  GenParticleInfo *gpi = dynamic_cast<GenParticleInfo *>(up);
  if (gpi == nullptr)
    G4Exception("SimG4Core/Notification", "mc001", FatalException, 
                "GenParticleInfoExtractor: user information in G4PrimaryParticle is not of GenParticleInfo type");
  return *gpi;
}
