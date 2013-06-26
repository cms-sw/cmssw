#ifndef MassKinematicConstraint_H
#define MassKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"

/**
 * Mass track constraint class. Provides a way to compute derivative
 * matrix and vector of values for mass constraint on the single trajectory.
 * This class does not have the multistate version, since vertex definition
 * needed to calculate a joint mass of several particles.
 * 
 * Kirill Prokofiev December 2002
 * Multitrack interface: July 2004
 */

class MassKinematicConstraint:public KinematicConstraint
{

public:

/**
 * Constructor with mass to be used as constraint value 
 */
 MassKinematicConstraint(const ParticleMass& m, const float sigma);
                                       
 
/**
 * Vector of values and matrix of derivatives
 * calculated at given 7xNumberOfStates point
 */ 
virtual std::pair<AlgebraicVector,AlgebraicVector> value(const AlgebraicVector& exPoint) const;

virtual std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const AlgebraicVector& exPoint) const;

/**
 * Vector of values and  matrix of derivatives calculated 
 * using current state parameters as expansion point
 */
virtual std::pair<AlgebraicVector, AlgebraicVector> value(const std::vector<RefCountedKinematicParticle> &par) const;

virtual std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const std::vector<RefCountedKinematicParticle> &par) const;

/**
 * Returns number of constraint equations used
 * for fitting. Method is relevant for proper NDF
 * calculations.
 */ 
virtual int numberOfEquations() const;

virtual AlgebraicVector deviations(int nStates) const;

virtual MassKinematicConstraint * clone() const
 {return new MassKinematicConstraint(*this);}

private:
  ParticleMass mass;
  AlgebraicVector dd; 
};

#endif
