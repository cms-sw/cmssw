#ifndef TwoTrackMassKinematicConstraint_H
#define TwoTrackMassKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"

/** 
 * Class implementing the total mass
 * of 2 tracks constraint. I.e. 2 first
 * particles out of four passed form 
 * a given mass
 *
 * Warning: the tracks to constraint 
 * should be 1st and 2nd from the
 * beginning of the vector.
 *
 */
                                               
class TwoTrackMassKinematicConstraint : public MultiTrackKinematicConstraint{

public:
 TwoTrackMassKinematicConstraint(ParticleMass& ms):mass(ms)
 {}


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
 
virtual TwoTrackMassKinematicConstraint * clone()const
{return new TwoTrackMassKinematicConstraint(*this);}

private:

 ParticleMass mass;

};
#endif
