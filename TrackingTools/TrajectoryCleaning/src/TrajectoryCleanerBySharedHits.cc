#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <map>
#include <vector>


class RecHitComparatorByPosition{
public:
  bool operator() (const  TransientTrackingRecHit* a, const TransientTrackingRecHit* b) const  {
    float xcut = 0.01;
    float ycut = 0.2;
    if (a->detUnit()<b->det()) return true;  
    if (b->detUnit()<a->det()) return false;  
    if (a->localPosition().x() < b->localPosition().x() - xcut)  return true;
    if (b->localPosition().x() < a->localPosition().x() - xcut) return false;
    return (a->localPosition().y() < b->localPosition().y() - ycut );
  }
};

void TrajectoryCleanerBySharedHits::clean( TrajectoryContainer & tc) const
{
  typedef vector<Trajectory>::iterator TI;
  typedef map<const TransientTrackingRecHit*,vector<TI>,RecHitComparatorByPosition> RecHitMap; 
  typedef map<Trajectory*,int,less<Trajectory*> > TrajMap;  // for each Trajectory it stores the number of shared hits
  RecHitMap theRecHitMap;

  // start filling theRecHit
  for (TrajectoryCleaner::TrajectoryIterator
	 it = tc.begin(); it != tc.end(); it++) {
    Trajectory::DataContainer pd = (*it).data();
    for (Trajectory::DataContainer::iterator im = pd.begin();
    	 im != pd.end(); im++) {
      const TransientTrackingRecHit* theRecHit = ((*im).recHit());
      if (theRecHit->isValid())
        theRecHitMap[theRecHit].push_back(it);
    }
  }
  // end filling theRecHit

  // for each trajectory fill theTrajMap

  for (TrajectoryCleaner::TrajectoryIterator
	 itt = tc.begin(); itt != tc.end(); itt++) {
    if((*itt).isValid()){  
      TrajMap theTrajMap;
      Trajectory::DataContainer pd = (*itt).data();
      for (Trajectory::DataContainer::iterator im = pd.begin();
	   im != pd.end(); im++) {
	const TransientTrackingRecHit* theRecHit = ((*im).recHit());	
        if (theRecHit->isValid()) {
	  const vector<TI>& hitTrajectories( theRecHitMap[theRecHit]);
	  for (vector<TI>::const_iterator ivec=hitTrajectories.begin(); 
	       ivec!=hitTrajectories.end(); ivec++) {
	    if (*ivec != itt){
	      if ((**ivec).isValid()){
		theTrajMap[&(**ivec)]++;
	      }
	    }
	  }
	}
      }
      //end filling theTrajMap

      // check for duplicated tracks
      if(!theTrajMap.empty() > 0){
	for(TrajMap::iterator imapp = theTrajMap.begin(); 
	    imapp != theTrajMap.end(); imapp++){
	  //          int nhit1 = (*itt).data().size();
	  //          int nhit2 = (*imapp).first->data().size();
          int nhit1 = (*itt).foundHits();
          int nhit2 = (*imapp).first->foundHits();
	  if((*imapp).second >= min(nhit1, nhit2)/2){
	    Trajectory* badtraj;
	    if (nhit1 != nhit2)
	      // select the shortest trajectory
	      badtraj = (nhit1 > nhit2) ?
		(*imapp).first : &(*itt);
	    else
	      // select the trajectory with less chi squared
	      badtraj = ((*imapp).first->chiSquared() > itt->chiSquared()) ?
		(*imapp).first : &(*itt);
	    badtraj->invalidate();  // invalidate this trajectory
	  }
	}
      } 
    }
  }

}

