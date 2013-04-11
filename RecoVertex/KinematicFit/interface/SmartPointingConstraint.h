#ifndef SmartPointingConstraint_H
#define SmartPointingConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"


/**
 *  Topological constraint making a momentum vector to point to
 *  the given location in space.
 *  Example: if b-meson momentum is reconstructed at b-meson decay position
 *  (secondary vertex), making reconstructed momentum  pointing the the primary 
 *  vertex
 *
 * Multiple track refit is not supported in current version
 *
 *  Kirill Prokofiev, March 2004
 *  MultiState version: July 2004
 */


class SmartPointingConstraint : public KinematicConstraint
{
 public:
 
  SmartPointingConstraint(const GlobalPoint& ref):refPoint(ref)
  {}

/**
 * Vector of values and  matrix of derivatives
 * calculated at given expansion 7xNumberOfStates point
 */ 
 virtual std::pair<AlgebraicVector, AlgebraicVector> value(const AlgebraicVector& exPoint) const;

 virtual std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const AlgebraicVector& exPoint) const;

/**
 * Vector of values and  matrix of derivatives calculated using current
 * state parameters as expansion point
 */
 virtual std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const std::vector<RefCountedKinematicParticle> &par) const;

 virtual std::pair<AlgebraicVector, AlgebraicVector> value(const std::vector<RefCountedKinematicParticle> &par) const;

 virtual AlgebraicVector deviations(int nStates) const;

/**
 * Returns number of constraint equations used
 * for fitting. Method is relevant for proper NDF
 * calculations.
 */ 
 virtual int numberOfEquations() const;

 virtual SmartPointingConstraint * clone() const
 {return new SmartPointingConstraint(*this);}
 
 private:

 std::pair<AlgebraicVector,AlgebraicVector> makeValue(const AlgebraicVector& exPoint)const ; 
 std::pair<AlgebraicMatrix, AlgebraicVector> makeDerivative(const AlgebraicVector& exPoint) const;
 
 GlobalPoint  refPoint;

};

#endif
