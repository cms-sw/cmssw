#include "SimG4Core/Application/interface/PrimaryTransformer.h"

#include "G4ParticleDefinition.hh"

PrimaryTransformer::PrimaryTransformer() : G4PrimaryTransformer()
{ unknownParticleDefined = false; }

PrimaryTransformer::~PrimaryTransformer() {}

G4ParticleDefinition * PrimaryTransformer::GetDefinition(G4PrimaryParticle * pp) 
{	       
    G4ParticleDefinition * partDef = pp->GetG4code();
    if(!partDef) partDef = particleTable->FindParticle(pp->GetPDGcode());
    if(unknownParticleDefined && ((!partDef)||partDef->IsShortLived())) partDef = unknown;
    return partDef;
}
