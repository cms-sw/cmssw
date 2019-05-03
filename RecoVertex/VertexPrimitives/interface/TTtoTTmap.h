#ifndef TTtoTTmap_H
#define TTtoTTmap_H

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include <map>

/** Map of (key = RecTrack,
 *          elt = map of(key = RecTrack, elt = covariance matrix))
 */

typedef std::map<reco::TransientTrack, AlgebraicMatrix33> TTmap;
typedef std::map<reco::TransientTrack, TTmap> TTtoTTmap;

#endif
