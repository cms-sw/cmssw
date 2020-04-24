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

class GsfMultipleScatteringUpdator final : public GsfMaterialEffectsUpdator {

public:

  /// constructor with explicit mass
  GsfMultipleScatteringUpdator(float mass) :
    GsfMaterialEffectsUpdator(mass,2) {}
  
  GsfMultipleScatteringUpdator* clone() const override
  {
    return new GsfMultipleScatteringUpdator(*this);
  }
  
  /// Computation: generates vectors of weights, means and standard deviations
  void compute (const TrajectoryStateOnSurface&, const PropagationDirection, Effect[]) const override;

  virtual size_t size() const { return 2;}
  

};

#endif
