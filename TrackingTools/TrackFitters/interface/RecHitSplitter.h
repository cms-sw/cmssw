#ifndef CD_RecHitSplitter_H_
#define CD_RecHitSplitter_H_

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/Common/interface/OwnVector.h"

/** Splits the matcher RecHits from the input container and 
 *  returns a container that has only unmatched hits.
 */

class RecHitSplitter {

public:

  RecHitSplitter() {}
  
  ~RecHitSplitter() {}

  edm::OwnVector<TransientTrackingRecHit> split(const edm::OwnVector<TransientTrackingRecHit>& hits) const;

private:
  
};

#endif //CD_RecHitSplitter_H_
