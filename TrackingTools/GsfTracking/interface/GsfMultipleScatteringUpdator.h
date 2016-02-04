#ifndef GsfMultipleScatteringUpdator_h_
#define GsfMultipleScatteringUpdator_h_

#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"

/** \class GsfMultipleScatteringUpdator
 *  Description of multiple scattering with two Gaussian
 *  components as described in HEPHY-PUB 724-99.
 *  Gaussians as a function of x/X0 are parametrized as polynomials.
 *  The mixture is parametrized as a function of the thickness,
 *  velocity and Xs=X0*h(Z).
 */

class GsfMultipleScatteringUpdator : public GsfMaterialEffectsUpdator {

public:
//   /// default constructor (mass from configurable)
//   GsfMultipleScatteringUpdator() :
//     GsfMaterialEffectsUpdator(),
//     theLastDz(0.),
//     theLastP(0.),
//     theLastPropDir(alongMomentum),
//     theLastRadLength(0.) {}
  /// constructor with explicit mass
  GsfMultipleScatteringUpdator(float mass) :
    GsfMaterialEffectsUpdator(mass),
    theLastDz(0.),
    theLastP(0.),
    theLastPropDir(alongMomentum),
    theLastRadLength(0.) {}
  
  virtual GsfMultipleScatteringUpdator* clone() const
  {
    return new GsfMultipleScatteringUpdator(*this);
  }
  
  
private:
  /// Computation: generates vectors of weights, means and standard deviations
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
