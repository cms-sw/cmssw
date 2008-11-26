#ifndef SimDataFormats_TrackingVertexContainer_h
#define SimDataFormats_TrackingVertexContainer_h

#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

#include <vector>

typedef std::vector<TrackingVertex>                TrackingVertexCollection;
typedef edm::Ref<TrackingVertexCollection>         TrackingVertexRef;
typedef edm::RefVector<TrackingVertexCollection>   TrackingVertexContainer;
typedef edm::RefVector<TrackingVertexCollection>   TrackingVertexRefVector;
typedef edm::RefProd<TrackingVertexCollection>     TrackingVertexRefProd;
typedef TrackingVertexRefVector::iterator   tv_iterator;


#endif
