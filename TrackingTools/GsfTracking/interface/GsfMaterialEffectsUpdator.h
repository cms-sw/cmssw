#ifndef GsfMaterialEffectsUpdator_h_
#define GsfMaterialEffectsUpdator_h_

#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include<cstdint>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

/**
 * Interface for adding material effects during propagation
 *   as a Gaussian mixture. Similar to MaterialEffectsUpdator,
 *   but returns MultiTrajectoryState.
 */
class GsfMaterialEffectsUpdator {  
public:
  typedef materialEffect::Covariance Covariance;
  typedef materialEffect::Effect Effect;
  typedef materialEffect::CovIndex CovIndex;


  /** Constructor with explicit mass hypothesis
   */
  GsfMaterialEffectsUpdator (float mass, uint32_t is ) :
    theMass(mass), m_size(is) {}

  virtual ~GsfMaterialEffectsUpdator () {}

  /** Updates TrajectoryStateOnSurface with material effects
   *    (momentum and covariance matrix are potentially affected.
   */
  virtual TrajectoryStateOnSurface updateState (const TrajectoryStateOnSurface& TSoS, 
						const PropagationDirection propDir) const;

  /** Particle mass assigned at construction.
   */
  inline float mass () const {
    return theMass;
  }

  virtual GsfMaterialEffectsUpdator* clone()  const = 0;

  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection, Effect effects[]) const = 0;

  size_t size() const { return m_size;}
protected:
  void resize(size_t is) { m_size=is;}

private:
  float theMass;
  uint32_t m_size;

};

#endif
