#ifndef CD_RecHitSorter_H_
#define CD_RecHitSorter_H_

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirection.h"

#include "DataFormats/Common/interface/OwnVector.h"

/** Sorts the RecHits along the PropagationDirection.
 */

class RecHitSorter {

public:

  RecHitSorter() {}

  ~RecHitSorter() {}

  edm::OwnVector<TransientTrackingRecHit> sortHits(const edm::OwnVector<TransientTrackingRecHit>& hits, 
			  const PropagationDirection& dir) const;

private:

};

#endif //TrajectoryStateWithArbitraryError
