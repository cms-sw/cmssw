#ifndef _CR_MULTIPLESCATTERINGUPDATOR_H_
#define _CR_MULTIPLESCATTERINGUPDATOR_H_

#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"

/** Adds effects from multiple scattering (standard Highland formula)
 *  to a trajectory state. Uses radiation length from medium properties.
 */
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
  /// default constructor (mass from configurable)
  MultipleScatteringUpdator() :
    MaterialEffectsUpdator(),
    theLastDz(0.),
    theLastP(0.),
    theLastPropDir(alongMomentum),
    theLastRadLength(0.) {}
  /// constructor with explicit mass
  MultipleScatteringUpdator( float mass ) :
    MaterialEffectsUpdator(mass),
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
  mutable float theLastDz;
  mutable float theLastP;
  mutable PropagationDirection theLastPropDir;
  mutable float theLastRadLength;
};

#endif
