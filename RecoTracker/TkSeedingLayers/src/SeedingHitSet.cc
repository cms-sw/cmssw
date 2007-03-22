#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

using namespace ctfseeding;

SeedingHitSet::SeedingHitSet(const Hits & hits)
  : theHits(hits)
{ 
// FIXME sort
}

void SeedingHitSet::add(const SeedingHit & aHit)
{
  // FIXME - put in correct place 
  theHits.push_back(aHit); 
}
