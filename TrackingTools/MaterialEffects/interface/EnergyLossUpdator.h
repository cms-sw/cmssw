#ifndef _CR_ENERGYLOSSUPDATOR_H_
#define _CR_ENERGYLOSSUPDATOR_H_

#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"
#include "Geometry/Vector/interface/LocalVector.h"

class MediumProperties;

/** Energy loss according to Bethe-Bloch + special treatment for electrons.
 * Adds effects from energy loss according to Bethe-Bloch formula
 *   without density effect. Assumes silicon as material.
 *   For electrons energy loss due to radiation added according 
 *   to formulae by Bethe & Heitler.
 */
class EnergyLossUpdator : public MaterialEffectsUpdator 
{
 public:
#ifndef CMS_NO_RELAXED_RETURN_TYPE
  virtual EnergyLossUpdator* clone() const
#else
  virtual MaterialEffectsUpdator* clone() const
#endif
  {
    return new EnergyLossUpdator(*this);
  }

public:
  /// default constructor (mass from configurable)
  EnergyLossUpdator() :
    MaterialEffectsUpdator(),
    theLastDz(0.),
    theLastP(0.),
    theLastPropDir(alongMomentum),
    theLastXi(0.) {}
  /// constructor with explicit mass value
  EnergyLossUpdator( float mass ) :
    MaterialEffectsUpdator(mass),
    theLastDz(0.),
    theLastP(0.),
    theLastPropDir(alongMomentum),
    theLastXi(0.) {}

private:
  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, 
			const PropagationDirection) const;
  // Internal routine for ionization acc. to Bethe-Bloch
  void computeBetheBloch (const LocalVector&, const MediumProperties&) const;
  // Internal routine for energy loss by electrons due to radiation
  void computeElectrons (const LocalVector&, const MediumProperties&,
			 const PropagationDirection) const;

protected:
  // check of arguments for use with cached values
  virtual bool newArguments (const TrajectoryStateOnSurface&, 
			     const PropagationDirection) const;
  // storage of arguments for later use of 
  virtual void storeArguments (const TrajectoryStateOnSurface&, 
			       const PropagationDirection) const;

private:  
  mutable float theLastDz;
  mutable float theLastP;
  mutable PropagationDirection theLastPropDir;
  mutable float theLastRl;
  mutable float theLastXi;
};

#endif
