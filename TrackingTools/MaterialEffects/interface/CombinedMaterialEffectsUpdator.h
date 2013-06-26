#ifndef _CR_COMBINEDMATERIALEFFECTSUPDATOR_H_
#define _CR_COMBINEDMATERIALEFFECTSUPDATOR_H_

/** \class CombinedMaterialEffectsUpdator
 *  Combines EnergyLossUpdator and MultipleScatteringUpdator.
 *  Adds effects from multiple scattering (via MultipleScatteringUpdator)
 *  and energy loss (via EnergyLossUpdator).
 *  Ported from ORCA.
 *
 *  \author todorov, cerati
 */

#include "TrackingTools/MaterialEffects/interface/MultipleScatteringUpdator.h"
#include "TrackingTools/MaterialEffects/interface/EnergyLossUpdator.h"
#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"
#include "FWCore/Utilities/interface/Visibility.h"

class CombinedMaterialEffectsUpdator GCC11_FINAL : public MaterialEffectsUpdator
{  
 public:
 virtual CombinedMaterialEffectsUpdator* clone() const {
    return new CombinedMaterialEffectsUpdator(*this);
 }

public:
  /// Specify assumed mass of particle for material effects.
  /// If ptMin > 0, then the rms muliple scattering angle will be calculated taking into account the uncertainty
  /// in the reconstructed track momentum. (By default, it is neglected). However, a lower limit on the possible
  /// value of the track Pt will be applied at ptMin, to avoid the rms multiple scattering becoming too big.
  CombinedMaterialEffectsUpdator(double mass, double ptMin = -1. ) :
    MaterialEffectsUpdator(mass),
    theMSUpdator(mass, ptMin),
    theELUpdator(mass) {}

  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection, Effect & effect) const;
  
 private:
  // objects used for calculations of multiple scattering and energy loss
  MultipleScatteringUpdator theMSUpdator;
  EnergyLossUpdator theELUpdator;
};

#endif
