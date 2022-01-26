#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

#include <iostream>
#include <cassert>

using Hit = SeedingHitSet::RecHit;
using HitPointer = SeedingHitSet::ConstRecHitPointer;

int main() {
  HitPointer hitOne = reinterpret_cast<HitPointer>(new SiPixelRecHit());
  HitPointer hitTwo = reinterpret_cast<HitPointer>(new MTDTrackingRecHit());
  HitPointer hitThree = reinterpret_cast<HitPointer>(new SiStripRecHit1D());
  HitPointer hitFour = reinterpret_cast<HitPointer>(new SiStripRecHit2D());
  HitPointer hitFive = reinterpret_cast<HitPointer>(new Phase2TrackerRecHit1D());

  HitPointer HIT_NULL = nullptr;

  std::vector<HitPointer> containerOne = {hitOne, hitTwo, hitThree, hitFour, hitFive};
  std::vector<HitPointer> containerTwo = {hitOne, hitTwo, hitThree, HIT_NULL, hitFive};
  std::vector<HitPointer> containerThree = {hitOne, HIT_NULL, hitThree, hitFour, hitFive};

  std::cout << " > Hit pointers " << std::endl;
  std::cout << " >> SiPixelRecHit at " << hitOne << std::endl;
  std::cout << " >> MTDTrackingRecHit at " << hitTwo << std::endl;
  std::cout << " >> SiStripRecHit1D at " << hitThree << std::endl;
  std::cout << " >> SiStripRecHit2D at " << hitFour << std::endl;
  std::cout << " >> Phase2TrackerRecHit1D at " << hitFive << std::endl;
  std::cout << std::endl;

  std::cout << "==> Testing vector constructor " << std::endl;
  SeedingHitSet vecOne(containerOne);
  assert(vecOne.size() == containerOne.size());
  SeedingHitSet vecTwo(containerTwo);
  assert(vecTwo.size() == 3);
  SeedingHitSet vecThree(containerThree);
  assert(vecThree.size() == 0);

  std::cout << "==> Testing braced constructor " << std::endl;
  SeedingHitSet listOne({hitThree, hitFive, hitFour});
  assert(listOne.size() == 3);
  SeedingHitSet listTwo({HIT_NULL, hitTwo, hitFour});
  assert(listTwo.size() == 0);
  SeedingHitSet listThree({hitTwo, hitFour, hitOne, hitFive, HIT_NULL});
  assert(listThree.size() == 4);

  std::cout << "==> Testing two hits constructor " << std::endl;
  SeedingHitSet twoHitsOne(hitOne, hitTwo);
  assert(twoHitsOne.size() == 2);
  SeedingHitSet twoHitsTwo(HIT_NULL, hitTwo);
  assert(twoHitsTwo.size() == 0);
  SeedingHitSet twoHitsThree(hitOne, HIT_NULL);
  assert(twoHitsThree.size() == 0);

  std::cout << "==> Testing two hits constructor " << std::endl;
  SeedingHitSet threeHitsOne(hitOne, hitTwo, hitThree);
  assert(threeHitsOne.size() == 3);
  SeedingHitSet threeHitsTwo(hitOne, hitTwo, HIT_NULL);
  assert(threeHitsTwo.size() == 2);
  SeedingHitSet threeHitsThree(hitOne, HIT_NULL, hitThree);
  assert(threeHitsThree.size() == 0);

  std::cout << "==> Testing four hits constructor " << std::endl;
  SeedingHitSet fourHitsOne(hitOne, hitTwo, hitThree, hitFour);
  assert(fourHitsOne.size() == 4);
  SeedingHitSet fourHitsTwo(hitOne, hitTwo, HIT_NULL, hitFour);
  assert(fourHitsTwo.size() == 2);
  SeedingHitSet fourHitsThree(hitOne, hitTwo, hitThree, HIT_NULL);
  assert(fourHitsThree.size() == 3);
  SeedingHitSet fourHitsFour(hitOne, HIT_NULL, hitThree, hitFour);
  assert(fourHitsFour.size() == 0);

  return 0;
}
