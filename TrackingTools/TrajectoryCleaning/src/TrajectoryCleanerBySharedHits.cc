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
        return (h1 == h2) || ((h1->geographicalId() == h2->geographicalId()) && (h1->hit()->sharesInput(h2->hit(), TrackingRecHit::some)));
    }
};
// Define a hash, i.e. a number that must be equal if hits are equal, and should be different if they're not
struct HashByDetId : std::unary_function<const TransientTrackingRecHit *, std::size_t> {
    std::size_t operator()(const TransientTrackingRecHit *hit) const { 
        boost::hash<uint32_t> hasher; 
        return hasher(hit->geographicalId().rawId());
    }
};

using RecHitMap = cmsutil::SimpleAllocHashMultiMap<const TransientTrackingRecHit*, Trajectory *, HashByDetId, EqualsBySharesInput>;
using TrajMap = cmsutil::UnsortedDumbVectorMap<Trajectory*, int>;

struct Maps {
  Maps() : theRecHitMap(128,256,1024){} // allocate 128 buckets, one row for 256 keys and one row for 512 values
  RecHitMap theRecHitMap;
  TrajMap theTrajMap;
};

thread_local Maps theMaps;
}

using namespace std;

void TrajectoryCleanerBySharedHits::clean( TrajectoryPointerContainer & tc) const
{
  if (tc.size() <= 1) return; // nothing to clean

  auto & theRecHitMap = theMaps.theRecHitMap;

  theRecHitMap.clear(10*tc.size());           // set 10*tc.size() active buckets
                                              // numbers are not optimized

  DEBUG_PRINT(std::cout << "Filling RecHit map" << std::endl);
  for (TrajectoryPointerContainer::iterator
	 it = tc.begin(); it != tc.end(); ++it) {
    DEBUG_PRINT(std::cout << "  Processing trajectory " << *it << " (" << (*it)->foundHits() << " valid hits)" << std::endl);
    const Trajectory::DataContainer & pd = (*it)->measurements();
    for (Trajectory::DataContainer::const_iterator im = pd.begin();
    	 im != pd.end(); im++) {
      const TransientTrackingRecHit* theRecHit = &(*(*im).recHit());
      if (theRecHit->isValid()) {
        DEBUG_PRINT(std::cout << "    Added hit " << theRecHit << " for trajectory " << *it << std::endl);
        theRecHitMap.insert(theRecHit, *it);
      }
    }
  }
  DEBUG_PRINT(theRecHitMap.dump());

  DEBUG_PRINT(std::cout << "Using RecHit map" << std::endl);
  // for each trajectory fill theTrajMap
  auto & theTrajMap = theMaps.theTrajMap; 
  for (TrajectoryCleaner::TrajectoryPointerIterator
	 itt = tc.begin(); itt != tc.end(); ++itt) {
    if((*itt)->isValid()){  
      DEBUG_PRINT(std::cout << "  Processing trajectory " << *itt << " (" << (*itt)->foundHits() << " valid hits)" << std::endl);
      theTrajMap.clear();
      const Trajectory::DataContainer & pd = (*itt)->measurements();
      for (Trajectory::DataContainer::const_iterator im = pd.begin();
	   im != pd.end(); ++im) {
	//RC const TransientTrackingRecHit* theRecHit = ((*im).recHit());
	const TransientTrackingRecHit* theRecHit = &(*(*im).recHit());
        if (theRecHit->isValid()) {
          DEBUG_PRINT(std::cout << "    Searching for overlaps on hit " << theRecHit << " for trajectory " << *itt << std::endl);
          for (RecHitMap::value_iterator ivec = theRecHitMap.values(theRecHit);
                ivec.good(); ++ivec) {
              if (*ivec != *itt){
                if ((*ivec)->isValid()){
                    theTrajMap[*ivec]++;
                }
              }
          }
	}
      }
      //end filling theTrajMap

      // check for duplicated tracks
      if(!theTrajMap.empty() > 0){
	for(TrajMap::iterator imapp = theTrajMap.begin(); 
	    imapp != theTrajMap.end(); ++imapp){
	  if((*imapp).second > 0 ){
	    int innerHit = 0;
	    if ( allowSharedFirstHit ) {
	      const TrajectoryMeasurement & innerMeasure1 = ( (*itt)->direction() == alongMomentum ) ? 
		(*itt)->firstMeasurement() : (*itt)->lastMeasurement();
	      const TransientTrackingRecHit* h1 = &(*(innerMeasure1).recHit());
	      const TrajectoryMeasurement & innerMeasure2 = ( (*imapp).first->direction() == alongMomentum ) ? 
		(*imapp).first->firstMeasurement() : (*imapp).first->lastMeasurement();
	      const TransientTrackingRecHit* h2 = &(*(innerMeasure2).recHit());
	      if ( (h1 == h2) || ((h1->geographicalId() == h2->geographicalId()) && 
				  (h1->hit()->sharesInput(h2->hit(), TrackingRecHit::some))) ) {
		innerHit = 1;
	      }
	    }
	    int nhit1 = (*itt)->foundHits();
	    int nhit2 = (*imapp).first->foundHits();
	    if( ((*imapp).second - innerHit) >= ( (min(nhit1, nhit2)-innerHit) * theFraction) ){
	      Trajectory* badtraj;
	      double score1 = validHitBonus_*nhit1 - missingHitPenalty_*(*itt)->lostHits() - (*itt)->chiSquared();
	      double score2 = validHitBonus_*nhit2 - missingHitPenalty_*(*imapp).first->lostHits() - (*imapp).first->chiSquared();
	      badtraj = (score1 > score2) ? (*imapp).first : *itt;
	      badtraj->invalidate();  // invalidate this trajectory
	    }
	  }
	}
      } 
    }
  }
}
