#ifndef PointingKinematicConstraint_H
#define PointingKinematicConstraint_H

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


class PointingKinematicConstraint : public KinematicConstraint
{
 public:
 
  PointingKinematicConstraint(const GlobalPoint& ref):refPoint(ref)
  {}

/**
 * Vector of values and  matrix of derivatives
 * calculated at given expansion 7xNumberOfStates point
 */ 
 std::pair<AlgebraicVector, AlgebraicVector> value(const AlgebraicVector& exPoint) const override;

 std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const AlgebraicVector& exPoint) const override;

/**
 * Vector of values and  matrix of derivatives calculated using current
 * state parameters as expansion point
 */
 std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const std::vector<RefCountedKinematicParticle> &par) const override;

 std::pair<AlgebraicVector, AlgebraicVector> value(const std::vector<RefCountedKinematicParticle> &par) const override;

 AlgebraicVector deviations(int nStates) const override;

/**
 * Returns number of constraint equations used
 * for fitting. Method is relevant for proper NDF
 * calculations.
 */ 
 int numberOfEquations() const override;

 PointingKinematicConstraint * clone() const override
 {return new PointingKinematicConstraint(*this);}
 
 private:

 std::pair<AlgebraicVector,AlgebraicVector> makeValue(const AlgebraicVector& exPoint)const ; 
 std::pair<AlgebraicMatrix, AlgebraicVector> makeDerivative(const AlgebraicVector& exPoint) const;
 
 GlobalPoint  refPoint;

};

#endif
