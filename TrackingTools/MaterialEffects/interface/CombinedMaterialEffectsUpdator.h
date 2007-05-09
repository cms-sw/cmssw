#ifndef _CR_COMBINEDMATERIALEFFECTSUPDATOR_H_
#define _CR_COMBINEDMATERIALEFFECTSUPDATOR_H_

/** \class CombinedMaterialEffectsUpdator
 *  Combines EnergyLossUpdator and MultipleScatteringUpdator.
 *  Adds effects from multiple scattering (via MultipleScatteringUpdator)
 *  and energy loss (via EnergyLossUpdator).
 *  Ported from ORCA.
 *
 *  $Date: 2007/05/09 13:21:30 $
 *  $Revision: 1.2.2.1 $
 *  \author todorov, cerati
 */

#include "TrackingTools/MaterialEffects/interface/MultipleScatteringUpdator.h"
#include "TrackingTools/MaterialEffects/interface/EnergyLossUpdator.h"
#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"

class CombinedMaterialEffectsUpdator : public MaterialEffectsUpdator
{  
 public:
#ifndef CMS_NO_RELAXED_RETURN_TYPE
  virtual CombinedMaterialEffectsUpdator* clone() const
#else
  virtual MaterialEffectsUpdator* clone() const
#endif
  {
    return new CombinedMaterialEffectsUpdator(*this);
  }

public:
  /// constructor with explicit mass value
  CombinedMaterialEffectsUpdator( float mass ) :
    MaterialEffectsUpdator(mass),
    theMSUpdator(mass),
    theELUpdator(mass) {}

 private:
  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection) const;
  
 private:
  // objects used for calculations of multiple scattering and energy loss
  MultipleScatteringUpdator theMSUpdator;
  EnergyLossUpdator theELUpdator;
};

#endif
