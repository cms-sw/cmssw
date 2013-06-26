#ifndef KinematicConstraint_H
#define KinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/**
 * Pure abstract base class to implement different types 
 * of constraints for single tracks.
 * Class caches the information about calculation of
 * of constraint equation derivatives and values at given
 * linearization 7-point (x,y,z,p_x,p_y,p_z,m)_0. Fitter
 * usually takes current parameters as the first step point
 * and the change it to the result of the first iteration.
 *
 * Kirill Prokofiev, December 2002
 * Change for multistate refit: July 2004
 */
 
 
class KinematicConstraint{


public:

/**
 *  Default constructor and destructor
 */

KinematicConstraint() {}

virtual ~KinematicConstraint() {}

/**
 * Methods returning the constraint derivative matrix and value.
 * The equation expansion is done at the 7-point specified by user:
 * (x,y,z,p_x,p_y,p_z,m)_0. In case of multiple state refit
 * vector should be of dimension 7xNumberOfStates
 */

virtual std::pair<AlgebraicVector, AlgebraicVector> value(const AlgebraicVector& exPoint) const = 0;

virtual std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const AlgebraicVector& exPoint) const = 0;


/**
 * Methods making value and derivative
 * matrix using current state parameters
 * as expansion 7-point. Constraint can be 
 * made equaly for single and multiple states
 */
virtual std::pair<AlgebraicVector, AlgebraicVector> value(const std::vector<RefCountedKinematicParticle> &par) const = 0;

virtual std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const std::vector<RefCountedKinematicParticle> &par) const = 0;

/**
 * Returns vector of sigma squared  associated to the KinematicParameters
 * of refitted particles 
 * Initial deviations are given by user for the constraining parameters
 * (mass, momentum components etc).
 * In case of multiple states exactly the same values are added to
 * every particle parameters
 */
virtual AlgebraicVector deviations(int nStates) const = 0;

/**
 * Returns an actual number of equations in 
 * particular constraint (corresponds to 
 * the number of strings in constraint derivative matrix,
 * for example)
 */
virtual int numberOfEquations() const = 0;

/**
 * Clone method
 */
virtual KinematicConstraint * clone() const = 0;

};

#endif
