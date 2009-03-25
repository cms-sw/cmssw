#ifndef ColinearityKinematicConstraint_H
#define ColinearityKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/** 
 * Consstraint to force the two tracks to be colinear (parallel)
 *
 * Warning: Since this constraint makes only sense with two tracks, two and only 
 * two tracks should be used in the fit.
 *
 */
                                               
class ColinearityKinematicConstraint : public MultiTrackKinematicConstraint{

public:
 ColinearityKinematicConstraint() {}


/**
 * Returns a vector of values of constraint
 * equations at the point where the input
 * particles are defined.
 */
virtual AlgebraicVector  value(const vector<KinematicState> states,
                        const GlobalPoint& point) const;


/**
 * Returns a matrix of derivatives of
 * constraint equations w.r.t. 
 * particle parameters
 */
virtual AlgebraicMatrix parametersDerivative(const vector<KinematicState> states,
                                      const GlobalPoint& point) const;

/**
 * Returns a matrix of derivatives of
 * constraint equations w.r.t. 
 * vertex position
 */
virtual AlgebraicMatrix positionDerivative(const vector<KinematicState> states,
                                    const GlobalPoint& point) const;

/**
 * Number of equations per track used for the fit
 */
virtual int numberOfEquations() const {return 2;}
 
virtual ColinearityKinematicConstraint * clone()const
{return new ColinearityKinematicConstraint(*this);}


};
#endif
