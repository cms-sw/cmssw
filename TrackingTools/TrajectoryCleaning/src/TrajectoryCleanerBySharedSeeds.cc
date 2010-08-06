#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedSeeds.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <map>
#include <vector>

using namespace std;

/*****************************************************************************/
bool TrajectoryCleanerBySharedSeeds::sameSeed(const TrajectorySeed & s1,const TrajectorySeed & s2) const
{
  if(s1.nHits()==0 && s2.nHits()==0) return false;
  if(s1.nHits() != s2.nHits()) return false;

  TrajectorySeed::range r1 = s1.recHits();
  TrajectorySeed::range r2 = s2.recHits();

  TrajectorySeed::const_iterator h1 = r1.first;
  TrajectorySeed::const_iterator h2 = r2.first;

  do
  {
    if(!(h1->sharesInput(&(*h2),TrackingRecHit::all)))
      return false;

    h1++; h2++;
  }
  while(h1 != s1.recHits().second && 
        h2 != s2.recHits().second);

  return true;
}

/*****************************************************************************/
void TrajectoryCleanerBySharedSeeds::clean(TrajectoryPointerContainer&) const
{
}

/*****************************************************************************/
void TrajectoryCleanerBySharedSeeds::clean
  (std::vector<Trajectory> & trajs) const
{
  if(trajs.size() == 0) return;

  // Best track
  unsigned int best = 0;

  // Track are assumed to come in seed blocks
  for(unsigned int actual = 1; actual < trajs.size(); actual++)
  {
    if(sameSeed(trajs[best].seed(), trajs[actual].seed()))
    {
      // Track to remove 
      unsigned int remove;

      // remove track with lower number of found hits, higher chi2
      if(trajs[best].foundHits() != trajs[actual].foundHits())
      {
        if(trajs[best].foundHits()  > trajs[actual].foundHits())
          remove = actual;
        else { remove = best; best = actual; }
      }
      else
      {
        if(trajs[best].chiSquared() < trajs[actual].chiSquared())
          remove = actual;
        else { remove = best; best = actual; }
      }

      trajs[remove].invalidate();
    }    
  }

  LogTrace("TrajectoryCleanerBySharedSeeds") << "  [TrajecCleaner] cleaned trajs : 1/" << trajs.size()
					     << " (with " << trajs[best].measurements().size() << " hits)" << std::endl;
}

