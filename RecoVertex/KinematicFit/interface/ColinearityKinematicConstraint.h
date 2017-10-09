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
virtual int numberOfEquations() const {return size;}
 
virtual ColinearityKinematicConstraint * clone()const
  {return new ColinearityKinematicConstraint(*this);}

private:
  ConstraintDim dimension;
  unsigned int size;

};
#endif
