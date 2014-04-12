#ifndef FourMomentumKinematicConstraint_H
#define FourMomentumKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"

/**
 * 4-Momentum track constraint class provides a way to compute the
 * matrix of derivatives and the vector of values for 4-Momentum
 * constraint on for given KinematicParticle. Current version does
 * not allow working with multiple tracks
 * 
 * Kirill Prokofiev March 2003
 * MultiState version: July 2004
 */

class FourMomentumKinematicConstraint : public KinematicConstraint
{

public:

/**
 * Constructor with desired 4-momentum vector  and 
 * vector of deviations to be used forcovariance matrix as
 * arguments
 */
 FourMomentumKinematicConstraint(const AlgebraicVector& momentum,
                                 const AlgebraicVector& deviation);

/**
 * Vector of values and  matrix of derivatives
 * calculated at given  7*NumberOfStates expansion point
 */  
virtual std::pair<AlgebraicVector,AlgebraicVector> value(const AlgebraicVector& exPoint) const;
 
virtual std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const AlgebraicVector& exPoint) const;


/**
 * Vector of values and matrix of derivatives calculated using 
 * current state as an expansion point
 */  
virtual std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const std::vector<RefCountedKinematicParticle> &par) const;

virtual std::pair<AlgebraicVector, AlgebraicVector> value(const std::vector<RefCountedKinematicParticle> &par) const;

 
/**
 * Returns number of constraint equations used for fitting. Method is relevant for proper NDF
 * calculations.
 */ 
virtual int numberOfEquations() const;

virtual AlgebraicVector deviations(int nStates) const;

virtual FourMomentumKinematicConstraint * clone() const
 {return new FourMomentumKinematicConstraint(*this);}

private:

AlgebraicVector mm;
AlgebraicVector dd;

};
#endif
