#ifndef CombinedKinematicConstraint_H
#define CombinedKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

/**
 * This class combines several user defined constraints (by expanding the vector d and the matrices D and E).
 * Usage:
 * Add each constraint to a std::vector<MultiTrackKinematicConstraint* >.
 * This vector has to be used in the constructor:
 * MultiTrackKinematicConstraint *combiC = new CombinedKinematicConstraint(std::vector<MultiTrackKinematicConstraint* >)
 * The produced object can be used by KinematicConstrainedVertexFitter.fit()
 *
 * Lars Perchalla, Philip Sauerland, Dec 2009
 */

class CombinedKinematicConstraint : public MultiTrackKinematicConstraint{
	
public:
	CombinedKinematicConstraint(const std::vector<MultiTrackKinematicConstraint* > &constraintVector):constraints(constraintVector){
		if(constraints.empty()) throw VertexException("CombinedKinematicConstraint::<1 constraints passed.");
	}
	
	/**
	 * Returns a vector of values of the combined constraint
	 * equations at the point where the input
	 * particles are defined.
	 */
	AlgebraicVector  value(const std::vector<KinematicState> &states, const GlobalPoint& point) const override;
	
	/**
	 * Returns a matrix of derivatives of the combined
	 * constraint equations w.r.t. 
	 * particle parameters
	 */
	AlgebraicMatrix parametersDerivative(const std::vector<KinematicState> &states, const GlobalPoint& point) const override;
	
	/**
	 * Returns a matrix of derivatives of
	 * constraint equations w.r.t. 
	 * vertex position
	 */
	AlgebraicMatrix positionDerivative(const std::vector<KinematicState> &states, const GlobalPoint& point) const override;
	
	/**
	 * Number of equations per track used for the combined fit
	 */
	int numberOfEquations() const override;
	
	CombinedKinematicConstraint * clone()const override
	{
		return new CombinedKinematicConstraint(*this);
	}
	
private:
	std::vector<MultiTrackKinematicConstraint* > constraints;
	
};
#endif
