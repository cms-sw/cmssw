#ifndef CD_RecHitSorter_H_
#define CD_RecHitSorter_H_

/** \class RecHitSorter
 *  Sorts the RecHits along the PropagationDirection. Ported from ORCA
 *
 *  $Date: 2007/05/09 12:56:07 $
 *  $Revision: 1.4.2.1 $
 *  \author todorov, cerati
 */

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

class RecHitSorter {

public:

  typedef TransientTrackingRecHit::ConstRecHitContainer        RecHitContainer;

  RecHitSorter() {}

  ~RecHitSorter() {}

  RecHitContainer sortHits(const RecHitContainer& hits, 
			   const PropagationDirection& dir) const;

private:

};

#endif 
