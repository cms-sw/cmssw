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
virtual pair<AlgebraicVector,AlgebraicVector> value(const AlgebraicVector& exPoint) const;

virtual pair<AlgebraicMatrix, AlgebraicVector> derivative(const AlgebraicVector& exPoint) const;

/**
 * Vector of values and  matrix of derivatives calculated using current
 * state parameters as expansion point
 */
virtual pair<AlgebraicMatrix, AlgebraicVector> derivative(const vector<RefCountedKinematicParticle> par) const;

virtual pair<AlgebraicVector, AlgebraicVector> value(const vector<RefCountedKinematicParticle> par) const;

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
