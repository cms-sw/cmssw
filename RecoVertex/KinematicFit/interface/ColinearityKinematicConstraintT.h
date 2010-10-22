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

// just while testing...            
#ifndef ColinearityKinematicConstraint2_H
namespace  colinearityKinematic {
  enum ConstraintDim {Phi=1, PhiTheta=2};
}
#endif

template<enum colinearityKinematic::ConstraintDim Dim>                                 
class ColinearityKinematicConstraintT : public MultiTrackKinematicConstraintT<int(Dim),2>{

private:
  double a_1;
  double a_2;

  AlgebraicVector7 p1;
  AlgebraicVector7 p2;

  GlobalPoint point;

public:

  ColinearityKinematicConstraintT() :
    dimension(Dim),
    size(dimension == colinearityKinematic::Phi ? 1 :2){}
  
  
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
   * Returns a vector of values of constraint
   * equations at the point where the input
   * particles are defined.
   */
  virtual ROOT::Math::SVector<double,int(Dim)>  value() const;
  
  
  /**
   * Returns a matrix of derivatives of
   * constraint equations w.r.t. 
   * particle parameters
   */
  virtual ROOT::Math::SMatrix<double,int(Dim),14> parametersDerivative() const;
  
  /**
   * Returns a matrix of derivatives of
   * constraint equations w.r.t. 
   * vertex position
   */
  virtual ROOT::Math::SMatrix<double,int(Dim),3> positionDerivative() const;
  
  /**
   * Number of equations per track used for the fit
   */
  virtual int numberOfEquations() const {return size;}
  
  virtual ColinearityKinematicConstraintT<Dim> * clone()const
  {return new ColinearityKinematicConstraintT<Dim>(*this);}
  
private:
  colinearityKinematic::ConstraintDim const dimension;
  unsigned int const size;

};


template<enum colinearityKinematic::ConstraintDim Dim>                                 
ROOT::Math::SVector<double,int(Dim)>   
ColinearityKinematicConstraintT<Dim>::value() const
{
  ROOT::Math::SVector<double,int(Dim)> res;


  double p1vx = p1(3) - a_1*(point.y() - p1(1));
  double p1vy = p1(4) + a_1*(point.x() - p1(0));
 
  double p2vx = p2(3) - a_2*(point.y() - p2(1));
  double p2vy = p2(4) + a_2*(point.x() - p2(0));
  

  // H_phi:
  res(0)  = atan2(p1vy,p1vx) - atan2(p2vy,p2vx);
  if ( res(0) >  M_PI ) res(1) -= 2.0*M_PI;
  if ( res(0) <= -M_PI ) res(1) += 2.0*M_PI;
  // H_theta:
  if (Dim==colinearityKinematic::PhiTheta) {  
    double pt1  = sqrt(p1(3)*p1(3)+p1(4)*p1(4));
    double pt2  = sqrt(p2(3)*p2(3)+p2(4)*p2(4));
    res(1)  = atan2(pt1,p1(5)) - atan2(pt2,p2(5));
    if ( res(1) >  M_PI ) res(2) -= 2.0*M_PI;
    if ( res(1) <= -M_PI ) res(2) += 2.0*M_PI;
  }
  
  return res;
}

template<enum colinearityKinematic::ConstraintDim Dim>                                 
ROOT::Math::SMatrix<double,int(Dim),14> 
ColinearityKinematicConstraintT<Dim>::parametersDerivative() const
{
  ROOT::Math::SMatrix<double,int(Dim),14> res;

  double p1vx = p1(3) - a_1*(point.y() - p1(1));
  double p1vy = p1(4) + a_1*(point.x() - p1(0));
  double k1 = 1.0/(p1vx*p1vx + p1vy*p1vy);

  double p2vx = p2(3) - a_2*(point.y() - p2(1));
  double p2vy = p2(4) + a_2*(point.x() - p2(0));
  double k2 = 1.0/(p2vx*p2vx + p2vy*p2vy);

  // H_phi:

  //x1 and x2 derivatives: 1st and 8th elements
  res(0,0) =  -k1*a_1*p1vx;
  res(0,7) =   k2*a_2*p2vx;

  //y1 and y2 derivatives: 2nd and 9th elements:
  res(0,1) = -k1*a_1*p1vy;
  res(0,8) =  k2*a_2*p2vy;

  //z1 and z2 components: 3d and 10th elmnets stay 0:
  res(0,2)  = 0.; res(1,10) = 0.;

  //px1 and px2 components: 4th and 11th elements:
  res(0,3)  = -k1*p1vy;
  res(0,10) =  k2*p2vy;

  //py1 and py2 components: 5th and 12 elements:
  res(0,5)  =  k1*p1vx;
  res(0,11) = -k2*p2vx;


  //pz1 and pz2 components: 6th and 13 elements:
  res(0,5)  = 0.; res(0,12) = 0.;
  //mass components: 7th and 14th elements:
  res(0,5)  = 0.; res(0,13) = 0.;

  if (Dim==colinearityKinematic::PhiTheta)  {
    double pt1 = sqrt(p1(3)*p1(3)+p1(4)*p1(4));
    double pTot1  = p1(3)*p1(3)+p1(4)*p1(4)+p1(5)*p1(5);
    double pt2  = sqrt(p2(3)*p2(3)+p2(4)*p2(4));
    double pTot2 = p2(3)*p2(3)+p2(4)*p2(4)+p2(5)*p2(5);
    
    // H_theta:
    //x1 and x2 derivatives: 1st and 8th elements
    res(1,0) =  0.; res(1,7) = 0.;

    //y1 and y2 derivatives: 2nd and 9th elements:
    res(1,1) = 0.; res(1,8) = 0.;

    //z1 and z2 components: 3d and 10th elmnets stay 0:
    res(1,2) = 0.; res(1,9) = 0.;

    res(1,3)  =  p1(3) * (p1(5)/(pTot1*pt1));
    res(1,10) =  p2(3) * (-p2(5)/(pTot2*pt2));

    //py1 and py2 components: 5th and 12 elements:
    res(1,4)  =  p1(4)  * (p1(5)/(pTot1*pt1));
    res(1,11) =  p2(4)  * (-p2(5)/(pTot2*pt2));

    //pz1 and pz2 components: 6th and 13 elements:
    res(1,5)  = - pt1/pTot1;
    res(1,12) =   pt2/pTot2;

    //mass components: 7th and 14th elements:
    res(1,6)  = 0.; res(1,13) = 0.;
  }
  return res;
}

template<enum colinearityKinematic::ConstraintDim Dim>                                 
ROOT::Math::SMatrix<double,int(Dim),3>  
ColinearityKinematicConstraintT<Dim>::positionDerivative() const
{

  ROOT::Math::SMatrix<double,int(Dim),3> res;

  double p1vx = p1(3) - a_1*(point.y() - p1(1));
  double p1vy = p1(4) + a_1*(point.x() - p1(0));
  double k1 = 1.0/(p1vx*p1vx + p1vy*p1vy);

  double p2vx = p2(3) - a_2*(point.y() - p2(1));
  double p2vy = p2(4) + a_2*(point.x() - p2(0));
  double k2 = 1.0/(p2vx*p2vx + p2vy*p2vy);
 
  // H_phi:

  // xv component
  res(0,0) = k1*a_1*p1vx - k2*a_2*p2vx;

  //yv component
  res(0,1) =  k1*a_1*p1vy - k2*a_2*p2vy;

  //zv component
  res(0,2) = 0.;

  // H_theta: no correlation with vertex position
  if (Dim==colinearityKinematic::PhiTheta) {
    res(1,0) = 0.;
    res(1,1) = 0.;
    res(1,2) = 0.;
  }

  return res;
}



#endif
