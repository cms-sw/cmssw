#ifndef GsfVertexUpdator_H
#define GsfVertexUpdator_H

#include "RecoVertex/VertexPrimitives/interface/VertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexWeightCalculator.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexMerger.h"

/**
 *  Vertex updator for the Gaussian Sum vertex filter.
 *  (c.f. Th.Speer & R. Fruewirth, Comp.Phys.Comm 174, 935 (2006) )
 */

class GsfVertexUpdator : public VertexUpdator<5> {
public:
  typedef CachingVertex<5>::RefCountedVertexTrack RefCountedVertexTrack;
  typedef VertexTrack<5>::RefCountedLinearizedTrackState RefCountedLinearizedTrackState;

  GsfVertexUpdator(bool limit = false, const GsfVertexMerger* merger = nullptr);
  /**
 *  Method to add a track to an existing CachingVertex
 *  An invalid vertex is returned in case of problems during the update.
 */

  CachingVertex<5> add(const CachingVertex<5>& oldVertex, const RefCountedVertexTrack track) const override;

  /**
 *  Method removing already used VertexTrack from existing CachingVertex
 *  This method is not yet implemented.
 */

  CachingVertex<5> remove(const CachingVertex<5>& oldVertex, const RefCountedVertexTrack track) const override;

  /**
 * Clone method
 */

  VertexUpdator<5>* clone() const override { return new GsfVertexUpdator(*this); }

private:
  typedef std::vector<VertexState> VSC;
  typedef std::vector<RefCountedLinearizedTrackState> LTC;
  typedef std::pair<double, double> WeightChi2Pair;
  typedef std::pair<VertexState, WeightChi2Pair> VertexComponent;
  typedef std::pair<VertexState, double> VertexChi2Pair;

  VertexComponent createNewComponent(const VertexState& oldVertex,
                                     const RefCountedLinearizedTrackState linTrack,
                                     float weight,
                                     int sign) const;

  VertexChi2Pair assembleVertexComponents(const std::vector<VertexComponent>& newVertexComponents) const;

  bool limitComponents;
  DeepCopyPointerByClone<GsfVertexMerger> theMerger;
  KalmanVertexUpdator<5> kalmanVertexUpdator;
  GsfVertexWeightCalculator theWeightCalculator;
};

#endif
