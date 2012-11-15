#ifndef GsfMaterialEffectsAdapter_H_
#define GsfMaterialEffectsAdapter_H_

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"

/** Make standard (single state) MaterialEffectsUpdator usable in
 *  the context of GSF.
 */
class GsfMaterialEffectsAdapter  GCC11_FINAL : public GsfMaterialEffectsUpdator 
{
  virtual GsfMaterialEffectsAdapter* clone() const
  {
    return new GsfMaterialEffectsAdapter(*this);
  }

public:


  GsfMaterialEffectsAdapter( const MaterialEffectsUpdator& aMEUpdator ) :
    GsfMaterialEffectsUpdator(aMEUpdator.mass(),1),
    theMEUpdator(aMEUpdator.clone()) {}

  ~GsfMaterialEffectsAdapter() {}

  
  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection, Effect[]) const;


private:  
  DeepCopyPointerByClone<MaterialEffectsUpdator> theMEUpdator;
};

#endif
