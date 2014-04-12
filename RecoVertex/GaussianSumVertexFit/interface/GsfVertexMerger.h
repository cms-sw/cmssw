#ifndef GsfVertexMerger_H
#define GsfVertexMerger_H

#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"

/** 
 * Class which controls the reduction of vertex state components after
 * a GSF update.
 */
class VertexState;

class GsfVertexMerger {
public:

  GsfVertexMerger(const edm::ParameterSet& pSet);
  ~GsfVertexMerger() {}

  CachingVertex<5> merge(const CachingVertex<5> & vertex) const;

  VertexState merge(const VertexState & vertex) const;

  GsfVertexMerger * clone() const {
    return new GsfVertexMerger(* this);
  }

private:
  DeepCopyPointerByClone< MultiGaussianStateMerger<3> > merger;
  unsigned int maxComponents;
};

#endif
