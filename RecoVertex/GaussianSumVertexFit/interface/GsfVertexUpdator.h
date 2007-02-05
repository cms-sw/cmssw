#ifndef GsfVertexUpdator_H
#define GsfVertexUpdator_H

#include "RecoVertex/VertexPrimitives/interface/VertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexWeightCalculator.h"

/**
 *  Vertex updator for the Gaussian Sum vertex filter.
 *  (c.f. R. Fruewirth et.al., Comp.Phys.Comm 100 (1997) 1
 */

class GsfVertexUpdator: public VertexUpdator {

public:

/**
 *  Method to add a track to an existing CachingVertex
 *
 */

   CachingVertex add(const CachingVertex & oldVertex,
        const RefCountedVertexTrack track) const;

/**
 *  Method removing already used VertexTrack from existing CachingVertex
 *
 */

   CachingVertex remove(const CachingVertex & oldVertex,
        const RefCountedVertexTrack track) const;

/**
 * Clone method
 */

   VertexUpdator * clone() const
   {
    return new GsfVertexUpdator(* this);
   }


private:

  typedef std::vector<VertexState> VSC;
  typedef std::vector<RefCountedLinearizedTrackState> LTC;
  typedef std::pair<double, double> WeightChi2Pair;
  typedef std::pair<VertexState, WeightChi2Pair> VertexComponent;
  typedef std::pair<VertexState, double> VertexChi2Pair;

  VertexComponent createNewComponent(const VertexState & oldVertex,
	 const RefCountedLinearizedTrackState linTrack, float weight, int sign) const;

  VertexChi2Pair assembleVertexComponents(
  	 const vector<VertexComponent> & newVertexComponents) const;

  KalmanVertexUpdator kalmanVertexUpdator;
  GsfVertexWeightCalculator theWeightCalculator;
};

#endif
