#ifndef TTtoTTmap_H
#define TTtoTTmap_H

#include <map>
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

/** Map of (key = RecTrack, 
 *          elt = map of(key = RecTrack, elt = covariance matrix))
 */

typedef std::map<reco::TransientTrack, AlgebraicMatrix33> TTmap;
typedef std::map<reco::TransientTrack, TTmap> TTtoTTmap;


#endif
