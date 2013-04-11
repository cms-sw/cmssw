#ifndef MultipleKinematicConstraint_H
#define MultipleKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"


/**
 * Class implementing constraint multiplication
 * for single or multiple track refit. Multiple track
 * refit does not include vertexing constraint: only refit 
 * of tarjectory parameters is usually done.
 */

class MultipleKinematicConstraint : public KinematicConstraint
{
public:

  MultipleKinematicConstraint()
  {em = true;}


/**
 * Vector of values and  matrix of derivatives
 * calculated at given 7xNumberOfStates linearization point
 */   
  std::pair<AlgebraicVector,AlgebraicVector> value(const AlgebraicVector& exPoint) const;
 
  std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const AlgebraicVector& exPoint) const;
  
/**
 * Vector of values and matrix of derivatives, 
 * calu=culated at linearization point of current
 * particle states
 */  
  std::pair<AlgebraicVector, AlgebraicVector> value(const std::vector<RefCountedKinematicParticle> &par) const;

  std::pair<AlgebraicMatrix, AlgebraicVector> derivative(const std::vector<RefCountedKinematicParticle> &par) const;

  int numberOfEquations() const;
  
/**
 * Method adding new constraint to the list of
 * existing ones.
 */  
  void addConstraint(KinematicConstraint * newConst) const; 

  AlgebraicVector deviations(int nStates) const;
  
  bool isEmpty() const
  {return em;}
  
  MultipleKinematicConstraint * clone() const
  {return new MultipleKinematicConstraint(*this);}
  
private:

  mutable std::vector<KinematicConstraint *>  cts;
  
  mutable bool em;
};
#endif
