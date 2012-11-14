#ifndef _CR_MATERIALEFFECTSUPDATOR_H_
#define _CR_MATERIALEFFECTSUPDATOR_H_

/** \class MaterialEffectsUpdator
 *  Interface for adding material effects during propagation.
 *  Updates to TrajectoryStateOnSurface are implemented 
 *  in this class.
 *  Ported from ORCA.
 *
 *  $Date: 2012/05/05 17:44:59 $
 *  $Revision: 1.14 $
 *  \author todorov, cerati
 */

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include "FWCore/Utilities/interface/Visibility.h"

class MaterialEffectsUpdator
{  
public:
  
  struct Effect {
    // Change in |p| from material effects.
    double deltaP=0;
    // Contribution to covariance matrix (in local co-ordinates) from material effects.
    AlgebraicSymMatrix55 deltaCov;
  };

  /** Constructor with explicit mass hypothesis
   */
  MaterialEffectsUpdator ( double mass );
  virtual ~MaterialEffectsUpdator ();

  /** Updates TrajectoryStateOnSurface with material effects
   *    (momentum and covariance matrix are potentially affected.
   */
  virtual TrajectoryStateOnSurface updateState (const TrajectoryStateOnSurface& TSoS, 
						const PropagationDirection propDir) const;

  /** Updates in place TrajectoryStateOnSurface with material effects
   *    (momentum and covariance matrix are potentially affected)
   *  Will return 'false' if the 'updateState' would have returned an invalid TSOS
   *  Note that the TSoS might be very well unchanged from this method 
   *  (just like 'updateState' can return the same TSOS)
   */
  virtual bool updateStateInPlace (TrajectoryStateOnSurface& TSoS, 
				   const PropagationDirection propDir) const;



  /** Particle mass assigned at construction.
   */
  inline double mass () const {
    return theMass;
  }

  virtual MaterialEffectsUpdator* clone()  const = 0;

  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection, Effect & effect) const = 0;
 
 private:
  double theMass;

};

#endif
