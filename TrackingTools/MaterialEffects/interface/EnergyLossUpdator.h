#ifndef _CR_ENERGYLOSSUPDATOR_H_
#define _CR_ENERGYLOSSUPDATOR_H_

/** \class EnergyLossUpdator
 *  Energy loss according to Bethe-Bloch + special treatment for electrons.
 *  Adds effects from energy loss according to Bethe-Bloch formula
 *  without density effect. Assumes silicon as material.
 *  For electrons energy loss due to radiation added according 
 *  to formulae by Bethe & Heitler.
 *  Ported from ORCA.
 *
 */

#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "FWCore/Utilities/interface/Visibility.h"

class MediumProperties;

class EnergyLossUpdator GCC11_FINAL : public MaterialEffectsUpdator 
{
 public:
  virtual EnergyLossUpdator* clone() const {
    return new EnergyLossUpdator(*this);
  }

public:
  EnergyLossUpdator( double mass ) :
    MaterialEffectsUpdator(mass) {}

  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, 
			const PropagationDirection, Effect & effect) const;

private:
  // Internal routine for ionization acc. to Bethe-Bloch
  void computeBetheBloch (const LocalVector&, const MediumProperties&, Effect & effect) const dso_internal;
  // Internal routine for energy loss by electrons due to radiation
  void computeElectrons (const LocalVector&, const MediumProperties&,
			 const PropagationDirection, Effect & effect) const dso_internal;

};

#endif
