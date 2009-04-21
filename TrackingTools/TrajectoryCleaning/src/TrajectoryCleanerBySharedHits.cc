#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/RecHitComparatorByPosition.h"
#include <map>
#include <vector>
#include <boost/unordered_map.hpp>

// Define when two rechits are equals
struct EqualsBySharesInput { 
    bool operator()(const TransientTrackingRecHit *h1, const TransientTrackingRecHit *h2) const {
        return (h1 == h2) || ((h1->geographicalId() == h2->geographicalId()) && (h1->hit()->sharesInput(h2->hit(), TrackingRecHit::all)));
    }
};
// Define a hash, i.e. a number that must be equal if hits are equal, and should be different if they're not
struct HashByDetId : std::unary_function<const TransientTrackingRecHit *, std::size_t> {
    std::size_t operator()(const TransientTrackingRecHit *hit) const { 
        boost::hash<uint32_t> hasher; 
        return hasher(hit->geographicalId().rawId());
    }
};

using namespace std;

void TrajectoryCleanerBySharedHits::clean( TrajectoryPointerContainer & tc) const
{
  typedef vector<Trajectory*>::iterator TI;
  typedef vector<TI> TIs;
  typedef boost::unordered_map<const TransientTrackingRecHit*, TIs, HashByDetId, EqualsBySharesInput> RecHitMap;
  //typedef map<Trajectory*,int,less<Trajectory*> > TrajMap;  // for each Trajectory it stores the number of shared hits
  typedef boost::unordered_map<Trajectory*, int> TrajMap;  // for each Trajectory it stores the number of shared hits
  RecHitMap theRecHitMap(20*tc.size()); 

  // start filling theRecHit
  for (TrajectoryCleaner::TrajectoryPointerIterator
	 it = tc.begin(); it != tc.end(); ++it) {
    const Trajectory::DataContainer & pd = (*it)->data();
    for (Trajectory::DataContainer::const_iterator im = pd.begin();
    	 im != pd.end(); im++) {
      //RC const TransientTrackingRecHit* theRecHit = ((*im).recHit());
      const TransientTrackingRecHit* theRecHit = &(*(*im).recHit());
      if (theRecHit->isValid()) {
        theRecHitMap[theRecHit].push_back(it);
      }
    }
  }

  // for each trajectory fill theTrajMap

  TrajMap theTrajMap(20);
  for (TrajectoryCleaner::TrajectoryPointerIterator
	 itt = tc.begin(); itt != tc.end(); ++itt) {
    if((*itt)->isValid()){  
      theTrajMap.clear();
      const Trajectory::DataContainer & pd = (*itt)->data();
      for (Trajectory::DataContainer::const_iterator im = pd.begin();
	   im != pd.end(); ++im) {
	//RC const TransientTrackingRecHit* theRecHit = ((*im).recHit());
	const TransientTrackingRecHit* theRecHit = &(*(*im).recHit());
        if (theRecHit->isValid()) {
          const TIs & hitTrajectories = theRecHitMap[theRecHit];
	  for (TIs::const_iterator ivec=hitTrajectories.begin(); 
	     ivec!=hitTrajectories.end(); ivec++) {
              if (*ivec != itt){
                if ((**ivec)->isValid()){
                    theTrajMap[**ivec]++;
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
	  //          int nhit1 = (*itt).data().size();
	  //          int nhit2 = (*imapp).first->data().size();
          int nhit1 = (*itt)->foundHits();
          int nhit2 = (*imapp).first->foundHits();
	  if((*imapp).second >= (min(nhit1, nhit2) * theFraction)){
	    Trajectory* badtraj;
	    if (nhit1 != nhit2)
	      // select the shortest trajectory
	      badtraj = (nhit1 > nhit2) ?
		(*imapp).first : *itt;
	    else
	      // select the trajectory with less chi squared
	      badtraj = ((*imapp).first->chiSquared() > (*itt)->chiSquared()) ?
		(*imapp).first : *itt;
	    badtraj->invalidate();  // invalidate this trajectory
	  }
	}
      } 
    }
  }
}

