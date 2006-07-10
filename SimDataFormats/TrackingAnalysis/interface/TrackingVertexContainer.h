#ifndef SimDataFormats_TrackingVertexContainer_h
#define SimDataFormats_TrackingVertexContainer_h

#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

#include <vector>
 
typedef std::vector<TrackingVertex>                TrackingVertexCollection;
typedef edm::Ref<TrackingVertexCollection>         TrackingVertexRef;
typedef edm::RefVector<TrackingVertexCollection>   TrackingVertexContainer;
 

#endif
