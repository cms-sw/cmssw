#ifndef GsfCombinedMaterialEffectsUpdator_h_
#define GsfCombinedMaterialEffectsUpdator_h_

#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

/** Combines two GsfMaterialEffectsUpdators (for multiple scattering
 *  and energy loss).
 */
class GsfCombinedMaterialEffectsUpdator GCC11_FINAL : public GsfMaterialEffectsUpdator
{  
 public:
  virtual GsfCombinedMaterialEffectsUpdator* clone() const
  {
    return new GsfCombinedMaterialEffectsUpdator(*this);
  }

public:
  /// Constructor from multiple scattering and energy loss updator
  GsfCombinedMaterialEffectsUpdator (GsfMaterialEffectsUpdator& msUpdator,
				     GsfMaterialEffectsUpdator& elUpdator);

  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection, Effect[]) const;

 
 private:
  // objects used for calculations of multiple scattering and energy loss
  DeepCopyPointerByClone<GsfMaterialEffectsUpdator> theMSUpdator;
  DeepCopyPointerByClone<GsfMaterialEffectsUpdator> theELUpdator;
};

#endif
