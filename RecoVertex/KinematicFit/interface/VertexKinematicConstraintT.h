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

  // to be optimized

  TrackCharge ch[2];
  GlobalVector mom[2];
  GlobalPoint pos[2];
  GlobalPoint point;
  double mfz;

public:

VertexKinematicConstraintT();
 
virtual ~VertexKinematicConstraintT();

  // initialize the constraint so it can precompute common qualtities to the three next call
  virtual void init(const std::vector<KinematicState>& states,
		    const GlobalPoint& point,  const GlobalVector& mf);


/**
 * Returns a vector of values of constraint
 * equations at the point where the input
 * particles are defined.
 */
 virtual ROOT::Math::SVector<double,4>  value() const;

/**
 * Returns a matrix of derivatives of
 * constraint equations w.r.t. 
 * particle parameters
 */
 virtual ROOT::Math::SMatrix<double,4,14> parametersDerivative() const;

/**
 * Returns a matrix of derivatives of
 * constraint equations w.r.t. 
 * vertex position
 */
 virtual ROOT::Math::SMatrix<double,4,3> positionDerivative() const;
/**
 * Number of equations per track used for the fit
 */
virtual int numberOfEquations() const;
 
virtual VertexKinematicConstraintT * clone()const
{return new VertexKinematicConstraintT(*this);}


private:

};
#endif
