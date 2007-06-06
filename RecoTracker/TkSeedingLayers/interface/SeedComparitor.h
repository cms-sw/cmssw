#ifndef RecoTracker_TkSeedingLayers_SeedComparitor_H
#define RecoTracker_TkSeedingLayers_SeedComparitor_H

/** \class SeedComparitor 
 * Base class for comparing a set of tracking seeds for compatibility.  This can
 * then be used to cleanup bad seeds.  Currently forseen are child classes that use
 * PixelStubs and Ferenc Sikler's similar objects for low Pt tracks.
 *  \author Aaron Dominguez (UNL)
 */
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

class SeedComparitor {
 public:
  virtual ~SeedComparitor() {}
  virtual bool compatible(const SeedingHitSet &hits) = 0;
};

#endif

