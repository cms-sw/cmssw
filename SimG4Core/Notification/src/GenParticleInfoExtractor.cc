#include "SimG4Core/Notification/interface/GenParticleInfoExtractor.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "G4PrimaryParticle.hh"

const GenParticleInfo &GenParticleInfoExtractor::operator()(const G4PrimaryParticle *p) const {
  G4VUserPrimaryParticleInformation *up = p->GetUserInformation();
  if (up == nullptr)
    throw SimG4Exception("GenParticleInfoExtractor: G4PrimaryParticle has no user information");
  GenParticleInfo *gpi = dynamic_cast<GenParticleInfo *>(up);
  if (gpi == nullptr)
    throw SimG4Exception("User information in G4PrimaryParticle is not of GenParticleInfo type");
  return *gpi;
}
