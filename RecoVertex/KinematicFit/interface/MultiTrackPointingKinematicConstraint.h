#ifndef MultiTrackPointingKinematicConstraint_H
#define MultiTrackPointingKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/**
 *  Topological constraint making a momentum vector to point to
 *  the given location in space.
 *  Example: if b-meson momentum is reconstructed at b-meson decay position
 *  (secondary vertex), making reconstructed momentum be in parallel to the link from primary 
 *  vertex to secondary vertex.
 *
 * 
 *  Kirill Prokofiev, March 2004
 *  MultiTrack version including propagation to linearization point: Lars Perchalla, Philip Sauerland, Dec 2009
 */
//mother constructed from daughters. including propagation in field.

class MultiTrackPointingKinematicConstraint : public MultiTrackKinematicConstraint
{
public:
	MultiTrackPointingKinematicConstraint(GlobalPoint& ref):refPoint(ref)
	{}
	
	/**
	 * Returns a vector of values of constraint
	 * equations at the point where the input
	 * particles are defined.
	 */
	AlgebraicVector value(const std::vector<KinematicState> &states, const GlobalPoint& point) const override;
	
	/**
	 * Returns a matrix of derivatives of
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
	 * Number of equations per track used for the fit
	 */
	int numberOfEquations() const override;
	
	MultiTrackPointingKinematicConstraint * clone()const override
	{
		return new MultiTrackPointingKinematicConstraint(*this);
	}
	
private:
	GlobalPoint refPoint;
	
};
#endif
