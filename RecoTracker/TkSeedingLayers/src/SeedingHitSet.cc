#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

using namespace ctfseeding;

SeedingHitSet::SeedingHitSet(const Hits & hits)
{ 
  for (Hits::const_iterator it=hits.begin(); it != hits.end(); ++it) theRecHits.push_back(*it);
}

void SeedingHitSet::add(const SeedingHit & aHit)
{
  theRecHits.push_back(aHit);
}
