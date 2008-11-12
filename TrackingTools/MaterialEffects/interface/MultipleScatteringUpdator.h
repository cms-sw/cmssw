#ifndef _CR_MULTIPLESCATTERINGUPDATOR_H_
#define _CR_MULTIPLESCATTERINGUPDATOR_H_

/** \class MultipleScatteringUpdator
 *  Adds effects from multiple scattering (standard Highland formula)
 *  to a trajectory state. Uses radiation length from medium properties.
 *  Ported from ORCA.
 *
 *  $Date: 2007/05/09 14:11:35 $
 *  $Revision: 1.3 $
 *  \author todorov, cerati
 */

#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"

class MultipleScatteringUpdator : public MaterialEffectsUpdator 
{
#ifndef CMS_NO_RELAXED_RETURN_TYPE
  virtual MultipleScatteringUpdator* clone() const
#else
  virtual MaterialEffectsUpdator* clone() const
#endif
  {
    return new MultipleScatteringUpdator(*this);
  }

public:
  /// Specify assumed mass of particle for material effects.
  /// If ptMin > 0, then the rms muliple scattering angle will be calculated taking into account the uncertainty
  /// in the reconstructed track momentum. (By default, it is neglected). However, a lower limit on the possible
  /// value of the track Pt will be applied at ptMin, to avoid the rms multiple scattering becoming too big.
  MultipleScatteringUpdator( float mass, float ptMin=-1. ) :
    MaterialEffectsUpdator(mass),
    thePtMin(ptMin),
    theLastDz(0.),
    theLastP(0.),
    theLastPropDir(alongMomentum),
    theLastRadLength(0.) {}
  /// destructor
  ~MultipleScatteringUpdator() {}
  /// reimplementation of deltaP (since always 0)
  virtual double deltaP (const TrajectoryStateOnSurface&, const PropagationDirection) const {
    return 0.;
  }

private:
  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection) const;

protected:
  // check of arguments for use with cached values
  virtual bool newArguments (const TrajectoryStateOnSurface&, const PropagationDirection) const;
  // storage of arguments for later use of 
  virtual void storeArguments (const TrajectoryStateOnSurface&, const PropagationDirection) const;

private:  

  float thePtMin;

  mutable float theLastDz;
  mutable float theLastP;
  mutable PropagationDirection theLastPropDir;
  mutable float theLastRadLength;
};

#endif
