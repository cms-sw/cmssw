#ifndef BackToBackKinematicConstraint_H
#define BackToBackKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"

/**
 * Class implementing the back-to-back geometry
 * constraints for 2 KinematicParticles. Current version
 * does not implement an actual back-to-back. 
 * Current class forces 2 tracks to have the opposite direction,
 * bud does not force them to emerge from the single point.
 *Coorect version to be implemented later.
 *
 * This is just
 * an illustrative piece of code, showing possible approach to 
 * constraint application on multiple tracks.
 * 
 * Kirill Prokofiev, July 2004
 */

class BackToBackKinematicConstraint:public KinematicConstraint
{
public:

 BackToBackKinematicConstraint() {}
 
 ~BackToBackKinematicConstraint() {}

/**
 * Derivatives and value calculated at given expansion point
 * Vector should always be of size 14 (2 particles)
 */
virtual std::pair<AlgebraicVector, AlgebraicVector> value(const AlgebraicVector& exPoint) const;

virtual std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const AlgebraicVector& exPoint) const;

/**
 * Derivatives and values calculated at expansion point, taken
 * at current state of input particles. Number of input particles
 * should be always equal to 2
 */
virtual std::pair<AlgebraicVector, AlgebraicVector> value(const std::vector<RefCountedKinematicParticle> &par) const;

virtual std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const std::vector<RefCountedKinematicParticle> &par) const;

virtual AlgebraicVector deviations(int nStates) const;

virtual int numberOfEquations() const;

virtual KinematicConstraint * clone() const;

private:


};
#endif
