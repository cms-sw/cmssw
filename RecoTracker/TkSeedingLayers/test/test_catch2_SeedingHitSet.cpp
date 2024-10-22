#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

#include "catch.hpp"

#include <iostream>

TEST_CASE("SeedingHitSet testing constructors", "[SeedingHitSet]") {
  using Hit = SeedingHitSet::RecHit;
  using HitPointer = SeedingHitSet::ConstRecHitPointer;

  HitPointer hitOne = new SiPixelRecHit();
  HitPointer hitTwo = new MTDTrackingRecHit();
  HitPointer hitThree = new SiStripRecHit1D();
  HitPointer hitFour = new SiStripRecHit2D();
  HitPointer hitFive = new Phase2TrackerRecHit1D();

  HitPointer HIT_NULL = nullptr;

  SECTION("Check vector constructor") {
    std::vector<HitPointer> containerOne = {hitOne, hitTwo, hitThree, hitFour, hitFive};
    std::vector<HitPointer> containerTwo = {hitOne, hitTwo, hitThree, HIT_NULL, hitFive};
    std::vector<HitPointer> containerThree = {hitOne, HIT_NULL, hitThree, hitFour, hitFive};

    SeedingHitSet vecOne(containerOne);
    REQUIRE(vecOne.size() == containerOne.size());
    SeedingHitSet vecTwo(containerTwo);
    REQUIRE(vecTwo.size() == 3);
    SeedingHitSet vecThree(containerThree);
    REQUIRE(vecThree.size() == 0);
  }

  SECTION("Check braced constructor") {
    SeedingHitSet listOne({hitThree, hitFive, hitFour});
    REQUIRE(listOne.size() == 3);
    SeedingHitSet listTwo({HIT_NULL, hitTwo, hitFour});
    REQUIRE(listTwo.size() == 0);
    SeedingHitSet listThree({hitTwo, hitFour, hitOne, hitFive, HIT_NULL});
    REQUIRE(listThree.size() == 4);
  }

  SECTION("Check two hits constructor") {
    SeedingHitSet twoHitsOne(hitOne, hitTwo);
    REQUIRE(twoHitsOne.size() == 2);
    SeedingHitSet twoHitsTwo(HIT_NULL, hitTwo);
    REQUIRE(twoHitsTwo.size() == 0);
    SeedingHitSet twoHitsThree(hitOne, HIT_NULL);
    REQUIRE(twoHitsThree.size() == 0);
  }

  SECTION("Check three hits constructor") {
    SeedingHitSet threeHitsOne(hitOne, hitTwo, hitThree);
    REQUIRE(threeHitsOne.size() == 3);
    SeedingHitSet threeHitsTwo(hitOne, hitTwo, HIT_NULL);
    REQUIRE(threeHitsTwo.size() == 2);
    SeedingHitSet threeHitsThree(hitOne, HIT_NULL, hitThree);
    REQUIRE(threeHitsThree.size() == 0);
  }
  SECTION("Check four hits constructor") {
    SeedingHitSet fourHitsOne(hitOne, hitTwo, hitThree, hitFour);
    REQUIRE(fourHitsOne.size() == 4);
    SeedingHitSet fourHitsTwo(hitOne, hitTwo, HIT_NULL, hitFour);
    REQUIRE(fourHitsTwo.size() == 2);
    SeedingHitSet fourHitsThree(hitOne, hitTwo, hitThree, HIT_NULL);
    REQUIRE(fourHitsThree.size() == 3);
    SeedingHitSet fourHitsFour(hitOne, HIT_NULL, hitThree, hitFour);
    REQUIRE(fourHitsFour.size() == 0);
  }
}
