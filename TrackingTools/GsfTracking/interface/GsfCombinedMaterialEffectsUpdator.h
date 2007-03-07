#ifndef GsfCombinedMaterialEffectsUpdator_h_
#define GsfCombinedMaterialEffectsUpdator_h_

#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

/** Combines two GsfMaterialEffectsUpdators (for multiple scattering
 *  and energy loss).
 */
class GsfCombinedMaterialEffectsUpdator : public GsfMaterialEffectsUpdator
{  
 public:
  virtual GsfCombinedMaterialEffectsUpdator* clone() const
  {
    return new GsfCombinedMaterialEffectsUpdator(*this);
  }

public:
//   /// Default constructor (mass from configurable)
//   GsfCombinedMaterialEffectsUpdator();
//   /// Constructor with explicit mass hypothesis
//   GsfCombinedMaterialEffectsUpdator( float mass );
  /// Constructor from multiple scattering and energy loss updator
  GsfCombinedMaterialEffectsUpdator (GsfMaterialEffectsUpdator& msUpdator,
				     GsfMaterialEffectsUpdator& elUpdator);

 private:
//   /// initialisation of individual updators
//   void createUpdators(const float);
  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection) const;
  
 private:
  // objects used for calculations of multiple scattering and energy loss
  DeepCopyPointerByClone<GsfMaterialEffectsUpdator> theMSUpdator;
  DeepCopyPointerByClone<GsfMaterialEffectsUpdator> theELUpdator;
};

#endif
