#ifndef ColinearityKinematicConstraint_H
#define ColinearityKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/** 
 * Consstraint to force the two tracks to be colinear (parallel), in 2D (phi) or 3D (phi-theta).
 *
 * Warning: Since this constraint makes only sense with two tracks, two and only 
 * two tracks should be used in the fit.
 *
 */
                                               
class ColinearityKinematicConstraint : public MultiTrackKinematicConstraint{

public:

 enum ConstraintDim {Phi, PhiTheta};

 ColinearityKinematicConstraint(ConstraintDim dim = Phi);


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
int numberOfEquations() const override {return size;}
 
ColinearityKinematicConstraint * clone()const override
  {return new ColinearityKinematicConstraint(*this);}

private:
  ConstraintDim dimension;
  unsigned int size;

};
#endif
