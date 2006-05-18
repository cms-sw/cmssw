#ifndef CD_RecHitSplitter_H_
#define CD_RecHitSplitter_H_

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/Common/interface/OwnVector.h"

/** Splits the matcher RecHits from the input container and 
 *  returns a container that has only unmatched hits.
 */

class RecHitSplitter {

public:

  typedef edm::OwnVector<const TransientTrackingRecHit>      RecHitContainer;

  RecHitSplitter() {}
  
  ~RecHitSplitter() {}

  RecHitContainer split(const RecHitContainer& hits) const;

private:
  
};

#endif //CD_RecHitSplitter_H_
