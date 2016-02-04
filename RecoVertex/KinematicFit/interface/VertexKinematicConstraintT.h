#ifndef VertexKinematicConstraintT_H
#define VertexKinematicConstraintT_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraintT.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/**
 * Class implementing the vertexing constraint 
 * for extended cartesian parametrization
 * (x,y,z,p_x,p_y,p_z,m). The equations and
 * derivatives in general follow the P.Avery's
 * "Applied Fitting Theory-VI" CBX 98-37
 */
class VertexKinematicConstraintT: public MultiTrackKinematicConstraintT<2,4>
{

private:

  typedef  MultiTrackKinematicConstraintT<2,4> super;

  // to be optimized

  double a_i[2];
  double novera[2], n[2],m[2],k[2], delta[2]; 
  GlobalVector mom[2];
  GlobalVector dpos[2];
  

public:

VertexKinematicConstraintT();
 
virtual ~VertexKinematicConstraintT();

  // initialize the constraint so it can precompute common qualtities to the three next call
  virtual void init(const std::vector<KinematicState>& states,
		    const GlobalPoint& point,  const GlobalVector& mf);


/**
 * Number of equations per track used for the fit
 */
virtual int numberOfEquations() const;
 
virtual VertexKinematicConstraintT * clone()const
{return new VertexKinematicConstraintT(*this);}


private:
/**
 * fills a vector of values of constraint
 * equations at the point where the input
 * particles are defined.
 */
 virtual void fillValue() const;

/**
 * fills a matrix of derivatives of
 * constraint equations w.r.t. 
 * particle parameters
 */
 virtual void fillParametersDerivative() const;

/**
 * fills a matrix of derivatives of
 * constraint equations w.r.t. 
 * vertex position
 */
 virtual void fillPositionDerivative() const;


private:

};
#endif
