#ifndef GsfMaterialEffectsAdapter_H_
#define GsfMaterialEffectsAdapter_H_

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"

/** Make standard (single state) MaterialEffectsUpdator usable in
 *  the context of GSF.
 */
class GsfMaterialEffectsAdapter  final : public GsfMaterialEffectsUpdator 
{
  GsfMaterialEffectsAdapter* clone() const override
  {
    return new GsfMaterialEffectsAdapter(*this);
  }

public:


  GsfMaterialEffectsAdapter( const MaterialEffectsUpdator& aMEUpdator ) :
    GsfMaterialEffectsUpdator(aMEUpdator.mass(),1),
    theMEUpdator(aMEUpdator.clone()) {}

  ~GsfMaterialEffectsAdapter() override {}

  
  // here comes the actual computation of the values
  void compute (const TrajectoryStateOnSurface&, const PropagationDirection, Effect[]) const override;


private:  
  DeepCopyPointerByClone<MaterialEffectsUpdator> theMEUpdator;
};

#endif
