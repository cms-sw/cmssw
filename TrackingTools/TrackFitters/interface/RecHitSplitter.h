#ifndef CD_RecHitSplitter_H_
#define CD_RecHitSplitter_H_

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

/** Splits the matcher RecHits from the input container and 
 *  returns a container that has only unmatched hits.
 */

class RecHitSplitter {

public:

  typedef TransientTrackingRecHit::ConstRecHitContainer  RecHitContainer;

  RecHitSplitter() {}
  
  ~RecHitSplitter() {}

  RecHitContainer split(const RecHitContainer& hits) const;

private:
  
};

#endif //CD_RecHitSplitter_H_
