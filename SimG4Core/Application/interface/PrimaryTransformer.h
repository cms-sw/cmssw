#ifndef SimG4Core_PrimaryTransformer_H
#define SimG4Core_PrimaryTransformer_H

#include "G4PrimaryTransformer.hh"

class PrimaryTransformer : public G4PrimaryTransformer
{
public:
    PrimaryTransformer();
    virtual ~PrimaryTransformer();
protected: 
    virtual G4ParticleDefinition * GetDefinition(G4PrimaryParticle * pp);
};

#endif 
