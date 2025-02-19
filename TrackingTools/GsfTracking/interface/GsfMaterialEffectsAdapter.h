#ifndef GsfMaterialEffectsAdapter_H_
#define GsfMaterialEffectsAdapter_H_

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"

/** Make standard (single state) MaterialEffectsUpdator usable in
 *  the context of GSF.
 */
class GsfMaterialEffectsAdapter : public GsfMaterialEffectsUpdator 
{
  virtual GsfMaterialEffectsAdapter* clone() const
  {
    return new GsfMaterialEffectsAdapter(*this);
  }

public:

//   GsfMaterialEffectsAdapter();

  GsfMaterialEffectsAdapter( const MaterialEffectsUpdator& aMEUpdator ) :
    GsfMaterialEffectsUpdator(aMEUpdator.mass()),
    theMEUpdator(aMEUpdator.clone()) {theWeights.push_back(1.);}

  ~GsfMaterialEffectsAdapter() {}

private:
  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection) const;

protected:
  // check of arguments for use with cached values
  virtual bool newArguments (const TrajectoryStateOnSurface&, const PropagationDirection) const
  {
    return true;
  }
  // storage of arguments for later use of 
  virtual void storeArguments (const TrajectoryStateOnSurface&, const PropagationDirection) const {}

private:  
  DeepCopyPointerByClone<MaterialEffectsUpdator> theMEUpdator;
};

#endif
