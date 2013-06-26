#ifndef MultiTrackMassKinematicConstraint_H
#define MultiTrackMassKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"

/**
 * Constraint to force some of the particles in the fit to have a certain invariant mass.
 */

class MultiTrackMassKinematicConstraint : public MultiTrackKinematicConstraint{

public:
  /**
   * Constructor
   * \param theMass the mass to constrain the states
   * \param nbrParticles the number of particles to use (in case more than that number are present
   * in the fit, the first will be used)
   */
  MultiTrackMassKinematicConstraint(const ParticleMass& theMass, const unsigned int nbrParticles)
	: mass(theMass), nPart(nbrParticles)
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
  virtual int numberOfEquations() const {return 1;}

  virtual MultiTrackMassKinematicConstraint * clone() const
  {return new MultiTrackMassKinematicConstraint(*this);}

private:

  const ParticleMass mass;
  const unsigned int nPart;

};
#endif
