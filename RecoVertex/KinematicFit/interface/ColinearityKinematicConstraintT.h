#ifndef ColinearityKinematicConstraintT_H
#define ColinearityKinematicConstraintT_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraintT.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include "RecoVertex/VertexPrimitives/interface/VertexException.h"


/** 
 * Constraint to force the two tracks to be colinear (parallel), in 2D (phi) or 3D (phi-theta).
 *
 * Warning: Since this constraint makes only sense with two tracks, two and only 
 * two tracks should be used in the fit.
 *
 */

namespace  colinearityKinematic {
  enum ConstraintDim {Phi=1, PhiTheta=2};
}

template<enum colinearityKinematic::ConstraintDim Dim>                                 
class ColinearityKinematicConstraintT : public MultiTrackKinematicConstraintT<int(Dim),2>{

private:
  double a_1;
  double a_2;

  AlgebraicVector7 p1;
  AlgebraicVector7 p2;

  GlobalPoint point;

public:

  typedef MultiTrackKinematicConstraintT<int(Dim),2> super;

  ColinearityKinematicConstraintT(){}
  
  
  // initialize the constraint so it can precompute common qualtities to the three next call
  void init(const std::vector<KinematicState>& states,
	    const GlobalPoint& ipoint,  const GlobalVector& fieldValue) {
    if(states.size()!=2) throw VertexException("ColinearityKinematicConstraint::<2 states passed");

    point = ipoint;

    a_1 = -states[0].particleCharge()*fieldValue.z();
    a_2 = -states[1].particleCharge()*fieldValue.z();

    p1 = states[0].kinematicParameters().vector();
    p2 = states[1].kinematicParameters().vector();
  }

  /**
   * Number of equations per track used for the fit
   */
  virtual int numberOfEquations() const {return Dim == colinearityKinematic::Phi ? 1 :2;}
  
  virtual ColinearityKinematicConstraintT<Dim> * clone()const
  {return new ColinearityKinematicConstraintT<Dim>(*this);}


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
   * Returns a matrix of derivatives of
   * constraint equations w.r.t. 
   * vertex position
   */
  virtual void fillPositionDerivative() const;
  
  
};


template<enum colinearityKinematic::ConstraintDim Dim>                                 
void ColinearityKinematicConstraintT<Dim>::fillValue() const {

  typename super::valueType & vl = super::vl();
 
  double p1vx = p1(3) - a_1*(point.y() - p1(1));
  double p1vy = p1(4) + a_1*(point.x() - p1(0));
 
  double p2vx = p2(3) - a_2*(point.y() - p2(1));
  double p2vy = p2(4) + a_2*(point.x() - p2(0));
  

  // H_phi:
  vl(0)  = atan2(p1vy,p1vx) - atan2(p2vy,p2vx);
  if ( vl(0) >  M_PI ) vl(0) -= 2.0*M_PI;
  if ( vl(0) <= -M_PI ) vl(0) += 2.0*M_PI;
  // H_theta:
  if (Dim==colinearityKinematic::PhiTheta) {  
    double pt1  = sqrt(p1(3)*p1(3)+p1(4)*p1(4));
    double pt2  = sqrt(p2(3)*p2(3)+p2(4)*p2(4));
    vl(1)  = atan2(pt1,p1(5)) - atan2(pt2,p2(5));
    if ( vl(1) >  M_PI ) vl(1) -= 2.0*M_PI;
    if ( vl(1) <= -M_PI ) vl(1) += 2.0*M_PI;
  }
}

template<enum colinearityKinematic::ConstraintDim Dim>                                 
void ColinearityKinematicConstraintT<Dim>::fillParametersDerivative() const {


  typename super::parametersDerivativeType & jac_d = super::jac_d(); 

  double p1vx = p1(3) - a_1*(point.y() - p1(1));
  double p1vy = p1(4) + a_1*(point.x() - p1(0));
  double k1 = 1.0/(p1vx*p1vx + p1vy*p1vy);

  double p2vx = p2(3) - a_2*(point.y() - p2(1));
  double p2vy = p2(4) + a_2*(point.x() - p2(0));
  double k2 = 1.0/(p2vx*p2vx + p2vy*p2vy);

  // H_phi:

  //x1 and x2 derivatives: 1st and 8th elements
  jac_d(0,0) =  -k1*a_1*p1vx;
  jac_d(0,7) =   k2*a_2*p2vx;

  //y1 and y2 derivatives: 2nd and 9th elements:
  jac_d(0,1) = -k1*a_1*p1vy;
  jac_d(0,8) =  k2*a_2*p2vy;

  //z1 and z2 components: 3d and 10th elmnets stay 0:
  jac_d(0,2)  = 0.; jac_d(0,9) = 0.;

  //px1 and px2 components: 4th and 11th elements:
  jac_d(0,3)  = -k1*p1vy;
  jac_d(0,10) =  k2*p2vy;

  //py1 and py2 components: 5th and 12 elements:
  jac_d(0,4)  =  k1*p1vx;
  jac_d(0,11) = -k2*p2vx;


  //pz1 and pz2 components: 6th and 13 elements:
  jac_d(0,5)  = 0.; jac_d(0,12) = 0.;
  //mass components: 7th and 14th elements:
  jac_d(0,6)  = 0.; jac_d(0,13) = 0.;

  if (Dim==colinearityKinematic::PhiTheta)  {
    double pt1 = sqrt(p1(3)*p1(3)+p1(4)*p1(4));
    double pTot1  = p1(3)*p1(3)+p1(4)*p1(4)+p1(5)*p1(5);
    double pt2  = sqrt(p2(3)*p2(3)+p2(4)*p2(4));
    double pTot2 = p2(3)*p2(3)+p2(4)*p2(4)+p2(5)*p2(5);
    
    // H_theta:
    //x1 and x2 derivatives: 1st and 8th elements
    jac_d(1,0) =  0.; jac_d(1,7) = 0.;

    //y1 and y2 derivatives: 2nd and 9th elements:
    jac_d(1,1) = 0.; jac_d(1,8) = 0.;

    //z1 and z2 components: 3d and 10th elmnets stay 0:
    jac_d(1,2) = 0.; jac_d(1,9) = 0.;

    jac_d(1,3)  =  p1(3) * (p1(5)/(pTot1*pt1));
    jac_d(1,10) =  p2(3) * (-p2(5)/(pTot2*pt2));

    //py1 and py2 components: 5th and 12 elements:
    jac_d(1,4)  =  p1(4)  * (p1(5)/(pTot1*pt1));
    jac_d(1,11) =  p2(4)  * (-p2(5)/(pTot2*pt2));

    //pz1 and pz2 components: 6th and 13 elements:
    jac_d(1,5)  = - pt1/pTot1;
    jac_d(1,12) =   pt2/pTot2;

    //mass components: 7th and 14th elements:
    jac_d(1,6)  = 0.; jac_d(1,13) = 0.;
  }
}

template<enum colinearityKinematic::ConstraintDim Dim>                                 
void ColinearityKinematicConstraintT<Dim>::fillPositionDerivative() const
{

  typename super::positionDerivativeType & jac_e = super::jac_e(); 

  double p1vx = p1(3) - a_1*(point.y() - p1(1));
  double p1vy = p1(4) + a_1*(point.x() - p1(0));
  double k1 = 1.0/(p1vx*p1vx + p1vy*p1vy);

  double p2vx = p2(3) - a_2*(point.y() - p2(1));
  double p2vy = p2(4) + a_2*(point.x() - p2(0));
  double k2 = 1.0/(p2vx*p2vx + p2vy*p2vy);
 
  // H_phi:

  // xv component
  jac_e(0,0) = k1*a_1*p1vx - k2*a_2*p2vx;

  //yv component
  jac_e(0,1) =  k1*a_1*p1vy - k2*a_2*p2vy;

  //zv component
  jac_e(0,2) = 0.;

  // H_theta: no correlation with vertex position
  if (Dim==colinearityKinematic::PhiTheta) {
    jac_e(1,0) = 0.;
    jac_e(1,1) = 0.;
    jac_e(1,2) = 0.;
  }
}



#endif
