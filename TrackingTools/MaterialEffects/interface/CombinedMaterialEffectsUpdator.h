#ifndef _CR_COMBINEDMATERIALEFFECTSUPDATOR_H_
#define _CR_COMBINEDMATERIALEFFECTSUPDATOR_H_

/** \class CombinedMaterialEffectsUpdator
 *  Combines EnergyLossUpdator and MultipleScatteringUpdator.
 *  Adds effects from multiple scattering (via MultipleScatteringUpdator)
 *  and energy loss (via EnergyLossUpdator).
 *  Ported from ORCA.
 *
 *  $Date: 2007/05/09 14:11:35 $
 *  $Revision: 1.3 $
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
  /// Specify assumed mass of particle for material effects.
  /// If ptMin > 0, then the rms muliple scattering angle will be calculated taking into account the uncertainty
  /// in the reconstructed track momentum. (By default, it is neglected). However, a lower limit on the possible
  /// value of the track Pt will be applied at ptMin, to avoid the rms multiple scattering becoming too big.
  CombinedMaterialEffectsUpdator( float mass, float ptMin = -1. ) :
    MaterialEffectsUpdator(mass),
    theMSUpdator(mass, ptMin),
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
