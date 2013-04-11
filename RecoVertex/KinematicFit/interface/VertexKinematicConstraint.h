#ifndef VertexKinematicConstraint_H
#define VertexKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/**
 * Class implementing the vertexing constraint 
 * for extended cartesian parametrization
 * (x,y,z,p_x,p_y,p_z,m). The equations and
 * derivatives in general follow the P.Avery's
 * "Applied Fitting Theory-VI" CBX 98-37
 */
class VertexKinematicConstraint:public MultiTrackKinematicConstraint
{
public:

VertexKinematicConstraint();
 
virtual ~VertexKinematicConstraint();

/**
 * Returns a vector of values of constraint
 * equations at the point where the input
 * particles are defined.
 */
virtual AlgebraicVector  value(const std::vector<KinematicState> &states,
                        const GlobalPoint& point) const;

/**
 * Returns a matrix of derivatives of
 * constraint equations w.r.t. 
 * particle parameters
 */
virtual AlgebraicMatrix parametersDerivative(const std::vector<KinematicState> &states,
                                      const GlobalPoint& point) const;

/**
 * Returns a matrix of derivatives of
 * constraint equations w.r.t. 
 * vertex position
 */
virtual AlgebraicMatrix positionDerivative(const std::vector<KinematicState> &states,
                                    const GlobalPoint& point) const;
/**
 * Number of equations per track used for the fit
 */
virtual int numberOfEquations() const;
 
virtual VertexKinematicConstraint * clone()const
{return new VertexKinematicConstraint(*this);}


private:

};
#endif
