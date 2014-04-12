#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"


AlgebraicVector  TwoTrackMassKinematicConstraint::value(const std::vector<KinematicState> &states,
                        const GlobalPoint& point) const
{ 
 if(states.size()<2) throw VertexException("TwoTrackMassKinematicConstraint::<2 states passed");

 AlgebraicVector res(1,0);
 
 double a_1 = -states[0].particleCharge() * states[0].magneticField()->inInverseGeV(states[0].globalPosition()).z();
 double a_2 = -states[1].particleCharge() * states[1].magneticField()->inInverseGeV(states[1].globalPosition()).z();

 AlgebraicVector7 p1 = states[0].kinematicParameters().vector();
 AlgebraicVector7 p2 = states[1].kinematicParameters().vector();
 
 double p1vx = p1(3) - a_1*(point.y() - p1(1));
 double p1vy = p1(4) + a_1*(point.x() - p1(0));
 double p1vz = p1(5);
 ParticleMass m1 = p1(6);
 
 double p2vx = p2(3) - a_2*(point.y() - p2(1));
 double p2vy = p2(4) + a_2*(point.x() - p2(0)); 
 double p2vz = p2(5);
 ParticleMass m2 = p2(6);
 
 double j_energy = sqrt(p1(3)*p1(3)+p1(4)*p1(4)+p1(5)*p1(5)+m1*m1)+
                  sqrt(p2(3)*p2(3)+p2(4)*p2(4)+p2(5)*p2(5)+m2*m2)  ;
 
 		       
 double j_m = (p1vx+p2vx)*(p1vx+p2vx) + (p1vy+p2vy)*(p1vy+p2vy) +
             (p1vz+p2vz)*(p1vz+p2vz);		        

 res(1)  = j_energy*j_energy - j_m - mass*mass;
 return res;
}			
			
AlgebraicMatrix TwoTrackMassKinematicConstraint::parametersDerivative(const std::vector<KinematicState> &states,
                                      const GlobalPoint& point) const
{
 int n_st = states.size();
 if(n_st<2) throw VertexException("TwoTrackMassKinematicConstraint::<2 states passed");

 AlgebraicMatrix res(1,n_st*7,0);
 
  double a_1 = -states[0].particleCharge() * states[0].magneticField()->inInverseGeV(states[0].globalPosition()).z();
  double a_2 = -states[1].particleCharge() * states[1].magneticField()->inInverseGeV(states[1].globalPosition()).z();
 
 AlgebraicVector7 p1 = states[0].kinematicParameters().vector();
 AlgebraicVector7 p2 = states[1].kinematicParameters().vector();
 
 double p1vx = p1(3) - a_1*(point.y() - p1(1));
 double p1vy = p1(4) + a_1*(point.x() - p1(0));
 double p1vz = p1(5);
 ParticleMass m1 = p1(6);
 
 double p2vx = p2(3) - a_2*(point.y() - p2(1));
 double p2vy = p2(4) + a_2*(point.x() - p2(0)); 
 double p2vz = p2(5);
 ParticleMass m2 = p2(6);
 
 double e1 = sqrt(p1(3)*p1(3)+p1(4)*p1(4)+p1(5)*p1(5)+m1*m1);
 double e2 = sqrt(p2(3)*p2(3)+p2(4)*p2(4)+p2(5)*p2(5)+m2*m2);


//x1 and x2 derivatives: 1st and 8th elements 
 res(1,1) = 2*a_1*(p2vy + p1vy);
 res(1,8) = 2*a_2*(p2vy + p1vy);

//y1 and y2 derivatives: 2nd and 9th elements:
 res(1,2) = -2*a_1*(p1vx + p2vx);
 res(1,9) = -2*a_2*(p2vx + p1vx);
 
//z1 and z2 components: 3d and 10th elmnets stay 0:
 res(1,3)  = 0.;
 res(1,10) = 0.;
 
//px1 and px2 components: 4th and 11th elements: 
 res(1,4)  = 2*(1+e2/e1)*p1(3) - 2*(p1vx + p2vx);
 res(1,11) = 2*(1+e1/e2)*p2(3) - 2*(p1vx + p2vx);

//py1 and py2 components: 5th and 12 elements:
  res(1,5)  = 2*(1+e2/e1)*p1(4) - 2*(p1vy + p2vy);
  res(1,12) = 2*(1+e1/e2)*p2(4) - 2*(p2vy + p1vy);
 
//pz1 and pz2 components: 6th and 13 elements:
 res(1,6)  = 2*(1+e2/e1)*p1(5)- 2*(p1vz + p2vz);
 res(1,13) = 2*(1+e1/e2)*p2(5)- 2*(p2vz + p1vz);
 
//mass components: 7th and 14th elements:
 res(1,7)  = 2*m1*(1+e2/e1);
 res(1,14) = 2*m2*(1+e1/e2);

  return res;
}			
				     				      
AlgebraicMatrix TwoTrackMassKinematicConstraint::positionDerivative(const std::vector<KinematicState> &states,
                                    const GlobalPoint& point) const
{
 AlgebraicMatrix res(1,3,0);
 if(states.size()<2) throw VertexException("TwoTrackMassKinematicConstraint::<2 states passed");

  double a_1 = -states[0].particleCharge() * states[0].magneticField()->inInverseGeV(states[0].globalPosition()).z();
  double a_2 = -states[1].particleCharge() * states[1].magneticField()->inInverseGeV(states[1].globalPosition()).z();
 
 AlgebraicVector7 p1 = states[0].kinematicParameters().vector();
 AlgebraicVector7 p2 = states[1].kinematicParameters().vector();
 
 double p1vx = p1(3) - a_1*(point.y() - p1(1));
 double p1vy = p1(4) + a_1*(point.x() - p1(0));
 
 double p2vx = p2(3) - a_2*(point.y() - p2(1));
 double p2vy = p2(4) + a_2*(point.x() - p2(0)); 
 
//xv component
 res(1,1) = -2*(p1vy + p2vy)*(a_1+a_2);

//yv component
 res(1,2) = 2*(p1vx + p2vx)*(a_1+a_2);

//zv component 
 res(1,3) = 0.; 
 
 return res;
}				    
				 				    
int TwoTrackMassKinematicConstraint::numberOfEquations() const
{return 1;}
