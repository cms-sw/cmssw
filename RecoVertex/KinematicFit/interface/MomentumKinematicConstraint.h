#ifndef MomentumKinematicConstraint_H
#define MomentumKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"


/**
 * Class constraining total 3-momentum of the particle (p_x,p_y,p_z)
 * This constraint can not be implemented on multiple particles,
 * without fitting the vertex. Current version supports one state
 * refit only.
 *
 * Kirill Prokofiev, October 2003
 * MultiState version: July 2004
 */
class MomentumKinematicConstraint : public KinematicConstraint
{

public:

/**
 * Constructor with the 4-momentum vector as
 * an argument
 */
MomentumKinematicConstraint(const AlgebraicVector& momentum, 
                            const AlgebraicVector& dev);

/**
 * Vector of values and  matrix of derivatives
 * calculated at given expansion 7xNumberOfStates point
 */ 
virtual std::pair<AlgebraicVector,AlgebraicVector> value(const AlgebraicVector& exPoint) const;

virtual std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const AlgebraicVector& exPoint) const;

/**
 * Vector of values and  matrix of derivatives calculated using current
 * state parameters as expansion point
 */
virtual std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const std::vector<RefCountedKinematicParticle> &par) const;

virtual std::pair<AlgebraicVector, AlgebraicVector> value(const std::vector<RefCountedKinematicParticle> &par) const;

virtual AlgebraicVector deviations(int nStates) const;

/**
 * Returns number of constraint equations used  for fitting.
 * Method is relevant for proper NDF calculations.
 */ 
virtual int numberOfEquations() const;

virtual MomentumKinematicConstraint * clone() const
  {return new MomentumKinematicConstraint(*this);}

private:

AlgebraicVector mm;
AlgebraicVector dd;

};


#endif
