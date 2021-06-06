#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/RecHitComparatorByPosition.h"

#include "TrackingTools/TrajectoryCleaning/src/OtherHashMaps.h"

//#define DEBUG_PRINT(X) X
#define DEBUG_PRINT(X)

namespace {

  // Define when two rechits are equals
  struct EqualsBySharesInput {
    bool operator()(const TransientTrackingRecHit *h1, const TransientTrackingRecHit *h2) const {
      return (h1 == h2) || ((h1->geographicalId() == h2->geographicalId()) &&
                            (h1->hit()->sharesInput(h2->hit(), TrackingRecHit::some)));
    }
  };
  // Define a hash, i.e. a number that must be equal if hits are equal, and should be different if they're not
  struct HashByDetId {
    std::size_t operator()(const TransientTrackingRecHit *hit) const {
      std::hash<uint32_t> hasher;
      return hasher(hit->geographicalId().rawId());
    }
  };

  using RecHitMap =
      cmsutil::SimpleAllocHashMultiMap<const TransientTrackingRecHit *, Trajectory *, HashByDetId, EqualsBySharesInput>;
  using TrajMap = cmsutil::UnsortedDumbVectorMap<Trajectory *, int>;

  struct Maps {
    Maps() : theRecHitMap(128, 256, 1024) {}  // allocate 128 buckets, one row for 256 keys and one row for 512 values
    RecHitMap theRecHitMap;
    TrajMap theTrajMap;
  };

  thread_local Maps theMaps;
}  // namespace

using namespace std;

void TrajectoryCleanerBySharedHits::clean(TrajectoryPointerContainer &tc) const {
  if (tc.size() <= 1)
    return;  // nothing to clean

  auto &theRecHitMap = theMaps.theRecHitMap;

  theRecHitMap.clear(10 * tc.size());  // set 10*tc.size() active buckets
                                       // numbers are not optimized

  DEBUG_PRINT(std::cout << "Filling RecHit map" << std::endl);
  for (auto const &it : tc) {
    DEBUG_PRINT(std::cout << "  Processing trajectory " << it << " (" << it->foundHits() << " valid hits)"
                          << std::endl);
    auto const &pd = it->measurements();
    for (auto const &im : pd) {
      auto theRecHit = &(*im.recHit());
      if (theRecHit->isValid()) {
        DEBUG_PRINT(std::cout << "    Added hit " << theRecHit << " for trajectory " << it << std::endl);
        theRecHitMap.insert(theRecHit, it);
      }
    }
  }
  DEBUG_PRINT(theRecHitMap.dump());

  DEBUG_PRINT(std::cout << "Using RecHit map" << std::endl);
  // for each trajectory fill theTrajMap
  auto &theTrajMap = theMaps.theTrajMap;
  for (auto const &itt : tc) {
    if (itt->isValid()) {
      DEBUG_PRINT(std::cout << "  Processing trajectory " << itt << " (" << itt->foundHits() << " valid hits)"
                            << std::endl);
      theTrajMap.clear();
      const Trajectory::DataContainer &pd = itt->measurements();
      for (auto const &im : pd) {
        auto theRecHit = &(*im.recHit());
        if (theRecHit->isValid()) {
          DEBUG_PRINT(std::cout << "    Searching for overlaps on hit " << theRecHit << " for trajectory " << itt
                                << std::endl);
          for (RecHitMap::value_iterator ivec = theRecHitMap.values(theRecHit); ivec.good(); ++ivec) {
            if (*ivec != itt) {
              if ((*ivec)->isValid()) {
                theTrajMap[*ivec]++;
              }
            }
          }
        }
      }
      //end filling theTrajMap

      auto score = [&](Trajectory const &t) -> float {
        // possible variant under study
        // auto ns = t.foundHits()-t.trailingFoundHits();
        //auto penalty =  0.8f*missingHitPenalty_;
        // return validHitBonus_*(t.foundHits()-0.2f*t.cccBadHits())  - penalty*t.lostHits() - t.chiSquared();
        // classical score
        return validHitBonus_ * t.foundHits() - missingHitPenalty_ * t.lostHits() - t.chiSquared();
      };

      // check for duplicated tracks
      if (!theTrajMap.empty() > 0) {
        for (auto const &imapp : theTrajMap) {
          if (imapp.second > 0) {  // at least 1 hits in common!!!
            int innerHit = 0;
            if (allowSharedFirstHit) {
              const TrajectoryMeasurement &innerMeasure1 =
                  (itt->direction() == alongMomentum) ? itt->firstMeasurement() : itt->lastMeasurement();
              const TransientTrackingRecHit *h1 = &(*(innerMeasure1).recHit());
              const TrajectoryMeasurement &innerMeasure2 = (imapp.first->direction() == alongMomentum)
                                                               ? imapp.first->firstMeasurement()
                                                               : imapp.first->lastMeasurement();
              const TransientTrackingRecHit *h2 = &(*(innerMeasure2).recHit());
              if ((h1 == h2) || ((h1->geographicalId() == h2->geographicalId()) &&
                                 (h1->hit()->sharesInput(h2->hit(), TrackingRecHit::some)))) {
                innerHit = 1;
              }
            }
            int nhit1 = itt->foundHits();
            int nhit2 = imapp.first->foundHits();
            if ((imapp.second - innerHit) >= ((min(nhit1, nhit2) - innerHit) * theFraction)) {
              Trajectory *badtraj;
              auto score1 = score(*itt);
              auto score2 = score(*imapp.first);
              badtraj = (score1 > score2) ? imapp.first : itt;
              badtraj->invalidate();  // invalidate this trajectory
            }
          }
        }
      }
    }
  }
}
