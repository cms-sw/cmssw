#ifndef TrackMap_H
#define TrackMap_H

#include <map>
#include "RecoVertex/VertexPrimitives/interface/RefCountedVertexTrack.h"

/** Map of (key = RefCountedVertexTrack, elt = covariance matrix)
 */

typedef std::map<RefCountedVertexTrack, AlgebraicMatrix> TrackMap;

#endif
