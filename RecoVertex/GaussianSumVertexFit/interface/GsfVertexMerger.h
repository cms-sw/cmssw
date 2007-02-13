#ifndef GsfVertexMerger_H
#define GsfVertexMerger_H

#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetAlgo/interface/DeepCopyPointerByClone.h"

/** 
 * Class which controls the reduction of vertex state components after
 * a GSF update.
 */
class CachingVertex;
class VertexState;

class GsfVertexMerger {
public:

  GsfVertexMerger(const edm::ParameterSet& pSet);
  ~GsfVertexMerger() {}

  CachingVertex merge(const CachingVertex & vertex) const;

  VertexState merge(const VertexState & vertex) const;

  GsfVertexMerger * clone() const {
    return new GsfVertexMerger(* this);
  }

private:
  DeepCopyPointerByClone<MultiGaussianStateMerger> merger;
  unsigned int maxComponents;
};

#endif
