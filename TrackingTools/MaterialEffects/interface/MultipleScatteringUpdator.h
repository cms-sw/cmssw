#ifndef _CR_MULTIPLESCATTERINGUPDATOR_H_
#define _CR_MULTIPLESCATTERINGUPDATOR_H_

/** \class MultipleScatteringUpdator
 *  Adds effects from multiple scattering (standard Highland formula)
 *  to a trajectory state. Uses radiation length from medium properties.
 *  Ported from ORCA.
 *
 *  \author todorov, cerati
 */

#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"
#include "FWCore/Utilities/interface/Visibility.h"

class MultipleScatteringUpdator GCC11_FINAL : public MaterialEffectsUpdator 
{
  virtual dso_export MultipleScatteringUpdator* clone() const {
    return new MultipleScatteringUpdator(*this);
  }

public:
  /// Specify assumed mass of particle for material effects.
  /// If ptMin > 0, then the rms muliple scattering angle will be calculated taking into account the uncertainty
  /// in the reconstructed track momentum. (By default, it is neglected). However, a lower limit on the possible
  /// value of the track Pt will be applied at ptMin, to avoid the rms multiple scattering becoming too big.
  MultipleScatteringUpdator(double mass, double ptMin=-1. ) :
    MaterialEffectsUpdator(mass),
    thePtMin(ptMin) {}
  /// destructor
  ~MultipleScatteringUpdator() {}


  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection, Effect & effect) const;


private:  

  double thePtMin;

};

#endif
