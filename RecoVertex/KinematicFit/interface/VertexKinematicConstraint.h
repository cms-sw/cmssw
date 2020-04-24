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
 
~VertexKinematicConstraint() override;

/**
 * Returns a vector of values of constraint
 * equations at the point where the input
 * particles are defined.
 */
AlgebraicVector  value(const std::vector<KinematicState> &states,
                        const GlobalPoint& point) const override;

/**
 * Returns a matrix of derivatives of
 * constraint equations w.r.t. 
 * particle parameters
 */
AlgebraicMatrix parametersDerivative(const std::vector<KinematicState> &states,
                                      const GlobalPoint& point) const override;

/**
 * Returns a matrix of derivatives of
 * constraint equations w.r.t. 
 * vertex position
 */
AlgebraicMatrix positionDerivative(const std::vector<KinematicState> &states,
                                    const GlobalPoint& point) const override;
/**
 * Number of equations per track used for the fit
 */
int numberOfEquations() const override;
 
VertexKinematicConstraint * clone()const override
{return new VertexKinematicConstraint(*this);}


private:

};
#endif
