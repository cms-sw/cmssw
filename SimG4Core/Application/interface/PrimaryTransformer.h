#ifndef SimG4Core_PrimaryTransformer_H
#define SimG4Core_PrimaryTransformer_H

#include "G4PrimaryTransformer.hh"

class PrimaryTransformer : public G4PrimaryTransformer {
public:
  PrimaryTransformer();
  ~PrimaryTransformer() override;

protected:
  G4ParticleDefinition* GetDefinition(G4PrimaryParticle* pp) override;
};

#endif
