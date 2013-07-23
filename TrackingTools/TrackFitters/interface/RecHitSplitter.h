#ifndef CD_RecHitSplitter_H_
#define CD_RecHitSplitter_H_

/** \class RecHitSplitter
 *  Splits the matcher RecHits from the input container and 
 *  returns a container that has only unmatched hits. Ported from ORCA
 *
 *  $Date: 2007/05/09 12:56:07 $
 *  $Revision: 1.3.2.1 $
 *  \author todorov, cerati
 */

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class RecHitSplitter {

public:

  typedef TransientTrackingRecHit::ConstRecHitContainer  RecHitContainer;

  RecHitSplitter() {}
  
  ~RecHitSplitter() {}

  RecHitContainer split(const RecHitContainer& hits) const;

private:
  
};

#endif //CD_RecHitSplitter_H_
