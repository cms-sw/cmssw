#ifndef TrackToTrackMap_H
#define TrackToTrackMap_H

#include <map>
#include "RecoVertex/VertexPrimitives/interface/RefCountedVertexTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TrackMap.h"

/** Map of (key = RefCountedVertexTrack, 
 *          elt = map of(key = RefCountedVertexTrack, elt = covariance matrix))
 */

typedef std::map<RefCountedVertexTrack, TrackMap> TrackToTrackMap;


#endif
