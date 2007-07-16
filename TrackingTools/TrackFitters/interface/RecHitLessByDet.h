#ifndef CD_RecHitLessByDet_H_
#define CD_RecHitLessByDet_H_

/** \class RecHitLessByDet
 *  A comparison operator for RecHits.
 *  Returns true if the hits are in the order in which they
 *  would be seen by an outgoing (or an incoming) track.
 *  Only meaningful for RecHits from the same trajectory. Ported from ORCA
 *
 *  $Date: 2007/05/09 14:17:57 $
 *  $Revision: 1.5 $
 *  \author todorov, cerati
 */

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class RecHitLessByDet {

  typedef TransientTrackingRecHit::ConstRecHitPointer RecHitPointer;
public:

  RecHitLessByDet(const PropagationDirection& dir) :
    theDirection(dir) {}

  ~RecHitLessByDet() {}

  bool operator()(const RecHitPointer& aHit, const RecHitPointer& bHit) const{

    return (theDirection == alongMomentum ? 
	    (aHit->surface().toGlobal(aHit->localPosition()).mag() < 
	     bHit->surface().toGlobal(bHit->localPosition()).mag() ) :
	    (aHit->surface().toGlobal(aHit->localPosition()).mag() >
	     bHit->surface().toGlobal(bHit->localPosition()).mag()) );
  }

private:
  
  PropagationDirection theDirection;

};
#endif //CD_RecHitLessByDet_H_
