#ifndef MultiTrackVertexLinkKinematicConstraint_H
#define MultiTrackVertexLinkKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/**
 *  This is an specialized version of MultiTrackVertexLinkKinematicConstraint.
 *  It constraints the sum of 4-vectors combined at a secondary vertex to be in parallel to the vertex link
 *  after considering the helix bend of the summed vector when propagating to the primary vertex.
 *
 *  Lars Perchalla, Philip Sauerland, July 2010
 */
//mother constructed from daughters. including propagation in field.

class MultiTrackVertexLinkKinematicConstraint : public MultiTrackKinematicConstraint
{
public:
	MultiTrackVertexLinkKinematicConstraint(GlobalPoint& ref):refPoint(ref)
	{}
	
	/**
	 * Returns a vector of values of constraint
	 * equations at the point where the input
	 * particles are defined.
	 */
	virtual AlgebraicVector value(const std::vector<KinematicState> &states, const GlobalPoint& point) const;
	
	/**
	 * Returns a matrix of derivatives of
	 * constraint equations w.r.t. 
	 * particle parameters
	 */
	virtual AlgebraicMatrix parametersDerivative(const std::vector<KinematicState> &states, const GlobalPoint& point) const;
	
	/**
	 * Returns a matrix of derivatives of
	 * constraint equations w.r.t. 
	 * vertex position
	 */
	virtual AlgebraicMatrix positionDerivative(const std::vector<KinematicState> &states, const GlobalPoint& point) const;
	
	/**
	 * Number of equations per track used for the fit
	 */
	virtual int numberOfEquations() const;
	
	virtual MultiTrackVertexLinkKinematicConstraint * clone()const
	{
		return new MultiTrackVertexLinkKinematicConstraint(*this);
	}
	
private:
	GlobalPoint refPoint;
	
};
#endif
