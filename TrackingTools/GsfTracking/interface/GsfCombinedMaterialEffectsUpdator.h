#ifndef GsfCombinedMaterialEffectsUpdator_h_
#define GsfCombinedMaterialEffectsUpdator_h_

#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

/** Combines two GsfMaterialEffectsUpdators (for multiple scattering
 *  and energy loss).
 */
class GsfCombinedMaterialEffectsUpdator final : public GsfMaterialEffectsUpdator
{  
 public:
  GsfCombinedMaterialEffectsUpdator* clone() const override
  {
    return new GsfCombinedMaterialEffectsUpdator(*this);
  }

public:
  /// Constructor from multiple scattering and energy loss updator
  GsfCombinedMaterialEffectsUpdator (GsfMaterialEffectsUpdator& msUpdator,
				     GsfMaterialEffectsUpdator& elUpdator);

  // here comes the actual computation of the values
  void compute (const TrajectoryStateOnSurface&, const PropagationDirection, Effect[]) const override;

 
 private:
  // objects used for calculations of multiple scattering and energy loss
  DeepCopyPointerByClone<GsfMaterialEffectsUpdator> theMSUpdator;
  DeepCopyPointerByClone<GsfMaterialEffectsUpdator> theELUpdator;
};

#endif
