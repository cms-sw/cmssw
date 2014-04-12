#include "RecoVertex/KinematicFit/interface/SimplePointingConstraint.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"


std::pair<AlgebraicVector, AlgebraicVector> SimplePointingConstraint::value(const AlgebraicVector& exPoint) const
{
 if(exPoint.num_row() ==0 ) throw VertexException("PointingKinematicConstraint::value requested for zero Linearization point");

//security check for extended cartesian parametrization 
 int inSize = exPoint.num_row(); 
 if((inSize%7) !=0) throw VertexException("PointingKinematicConstraint::linearization point has a wrong dimension");
 int nStates = inSize/7;
 if(nStates != 1) throw VertexException("PointingKinematicConstraint::Current version does not support the multistate refit");
 
 AlgebraicVector lPar = exPoint;
 AlgebraicVector vl(2,0);
 
//vector of values 1x2  for given particle
 AlgebraicVector lValue = makeValue(lPar).first;
 vl(1) =lValue(1);
 vl(2) =lValue(2);
 return std::pair<AlgebraicVector,AlgebraicVector>(vl,lPar); 
}

std::pair<AlgebraicMatrix, AlgebraicVector> SimplePointingConstraint::derivative(const AlgebraicVector& exPoint) const
{
 if(exPoint.num_row() ==0 ) throw VertexException("PointingKinematicConstraint::value requested for zero Linearization point");

//security check for extended cartesian parametrization 
 int inSize = exPoint.num_row(); 
 if((inSize%7) !=0) throw VertexException("PointingKinematicConstraint::linearization point has a wrong dimension");
 int nStates = inSize/7;
 if(nStates != 1) throw VertexException("PointingKinematicConstraint::Current version does not support the multistate refit");
 AlgebraicVector lPar = exPoint;

//2x7 derivative matrix for given particle
 AlgebraicMatrix lDeriv = makeDerivative(lPar).first;
 AlgebraicMatrix dr(2,7,0);
 dr.sub(1,1,lDeriv);
 return std::pair<AlgebraicMatrix,AlgebraicVector>(dr,lPar);
}

std::pair<AlgebraicMatrix, AlgebraicVector> SimplePointingConstraint::derivative(const std::vector<RefCountedKinematicParticle> &par) const
{
 int nStates = par.size();
 if(nStates == 0) throw VertexException("PointingKinematicConstraint::Empty vector of particles passed");
 if(nStates != 1) throw VertexException("PointingKinematicConstraint::Current version does not support the multistate refit");
 
 AlgebraicMatrix dr(2,7,0);
 AlgebraicVector lPoint = asHepVector<7>(par.front()->currentState().kinematicParameters().vector());

//2x7 derivative matrix for given state  
 AlgebraicMatrix lDeriv = makeDerivative(lPoint).first;
 dr.sub(1,1,lDeriv);
// cout<<"Derivative returned: "<<dr<<endl;
// cout<<"For the value: "<<lPoint<<endl;
 return std::pair<AlgebraicMatrix,AlgebraicVector>(dr,lPoint);
}

std::pair<AlgebraicVector, AlgebraicVector> SimplePointingConstraint::value(const std::vector<RefCountedKinematicParticle> &par) const
{ 
 int nStates = par.size();
 if(nStates == 0) throw VertexException("PointingKinematicConstraint::Empty vector of particles passed");
 if(nStates != 1) throw VertexException("PointingKinematicConstraint::Current version does not support the multistate refit");
 AlgebraicVector vl(2,0);
 AlgebraicVector lPoint = asHepVector<7>(par.front()->currentState().kinematicParameters().vector());
 vl(1) = makeValue(lPoint).first(1);
 vl(2) = makeValue(lPoint).first(2);
// cout<<"Value returned: "<<vl<<endl;
// cout<<"For the point: "<<lPoint<<endl;
 
 return std::pair<AlgebraicVector,AlgebraicVector>(vl,lPoint);
}
 
AlgebraicVector SimplePointingConstraint::deviations(int nStates) const
{return AlgebraicVector(7*nStates,0);}
 
int SimplePointingConstraint::numberOfEquations() const
{return 2;}
 
std::pair<AlgebraicVector,AlgebraicVector> SimplePointingConstraint::makeValue(const AlgebraicVector& exPoint)const 
{ 
// cout<<"Make value called"<<endl;
 AlgebraicVector vl(2,0);
 AlgebraicVector point = exPoint;
 double dx = point(1) - refPoint.x();
 double dy = point(2) - refPoint.y();
 double dz = point(3) - refPoint.z();
 double px = point(4);
 double py = point(5); 
 double pz = point(6);


//half angle solution: sin((alpha - betha)/2)
 double cos_phi_p = px/sqrt(px*px + py*py);
 double cos_phi_x = dx/sqrt(dx*dx + dy*dy);
// cout<<"mom cos phi"<<cos_phi_p<<endl;
// cout<<"x cos phi"<<cos_phi_x<<endl;
 
 
 double cos_theta_p = sqrt(px*px + py*py)/sqrt(px*px + py*py + pz*pz); 
 double cos_theta_x = sqrt(dx*dx + dy*dy)/sqrt(dx*dx + dy*dy + dz*dz);
 
 float feq = sqrt((1-cos_phi_p)*(1+cos_phi_x)) - sqrt((1+cos_phi_p)*(1-cos_phi_x));
 float seq = sqrt((1-cos_theta_p)*(1+cos_theta_x)) - sqrt((1+cos_theta_p)*(1-cos_theta_x));
 
// cout<<"First factor: "<<feq/2<<endl;
// cout<<"Second factor: "<<seq/2<<endl;
 
 vl(1) = feq/2;
 vl(2) = seq/2;

// cout<<"Value "<<vl<<endl;
//half angle corrected
// vl(1) = (sin_x/(1+cos_x)) - (sin_p/(1+cos_p));
// vl(2) = (sin_xt/(1+cos_xt)) - (sin_pt/(1+cos_pt));
 return std::pair<AlgebraicVector,AlgebraicVector>(vl,point);
}

std::pair<AlgebraicMatrix, AlgebraicVector> SimplePointingConstraint::makeDerivative(const AlgebraicVector& exPoint) const
{ 
 AlgebraicMatrix dr(2,7,0);
 AlgebraicVector point = exPoint;
 double dx = point(1) - refPoint.x();
 double dy = point(2) - refPoint.y();
 double dz = point(3) - refPoint.z();
 double px = point(4);
 double py = point(5); 
 double pz = point(6);
 

//half angle solution
//d/dx_i
 dr(1,1) = (sqrt((1 + dx/sqrt(pow(dx,2) + pow(dy,2)))*(1 - px/sqrt(pow(px,2) + pow(py,2)))) - 
            sqrt((1 - dx/sqrt(pow(dx,2) + pow(dy,2)))*(1 + px/sqrt(pow(px,2) + pow(py,2)))))/2.;
								  
 dr(1,2) = (((-(pow(dx,2)/pow(pow(dx,2) + pow(dy,2),1.5)) + 1/sqrt(pow(dx,2) + pow(dy,2)))*
           (1 - px/sqrt(pow(px,2) + pow(py,2))))/
           (2.*sqrt((1 + dx/sqrt(pow(dx,2) + pow(dy,2)))*(1 - px/sqrt(pow(px,2) + pow(py,2))))) - 
           ((pow(dx,2)/pow(pow(dx,2) + pow(dy,2),1.5) - 1/sqrt(pow(dx,2) + pow(dy,2)))*
           (1 + px/sqrt(pow(px,2) + pow(py,2))))/
           (2.*sqrt((1 - dx/sqrt(pow(dx,2) + pow(dy,2)))*(1 + px/sqrt(pow(px,2) + pow(py,2))))))/2.;
 
 
 dr(1,3) = 0;
 
//d/dp_i  
//debug: x->p index xhange in denominator
 dr(1,4) = (-(dx*dy*(1 - px/sqrt(pow(px,2) + pow(py,2))))/
           (2.*pow(pow(dx,2) + pow(dy,2),1.5)*
           sqrt((1 + dx/sqrt(pow(dx,2) + pow(dy,2)))*(1 - px/sqrt(pow(px,2) + pow(py,2))))) - 
           (dx*dy*(1 + px/sqrt(pow(px,2) + pow(py,2))))/
           (2.*pow(pow(dx,2) + pow(dy,2),1.5)*
           sqrt((1 - dx/sqrt(pow(dx,2) + pow(dy,2)))*(1 + px/sqrt(pow(px,2) + pow(py,2))))))/2.;
	
			   
 dr(1,5) = (((1 + dx/sqrt(pow(dx,2) + pow(dy,2)))*px*py)/
           (2.*pow(pow(px,2) + pow(py,2),1.5)*
           sqrt((1 + dx/sqrt(pow(dx,2) + pow(dy,2)))*(1 - px/sqrt(pow(px,2) + pow(py,2))))) + 
           ((1 - dx/sqrt(pow(dx,2) + pow(dy,2)))*px*py)/
           (2.*pow(pow(px,2) + pow(py,2),1.5)*
           sqrt((1 - dx/sqrt(pow(dx,2) + pow(dy,2)))*(1 + px/sqrt(pow(px,2) + pow(py,2))))))/2.;
 
 
 
 dr(1,6) = 0;
 dr(1,7) = 0; 

//2nd equation
//d/dx_i

 dr(2,1) =(((-((dx*sqrt(pow(dx,2) + pow(dy,2)))/pow(pow(dx,2) + pow(dy,2) + pow(dz,2),1.5)) + 
          dx/(sqrt(pow(dx,2) + pow(dy,2))*sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2))))*
          (1 - sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))/
          (2.*sqrt((1 + sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
          (1 - sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))) - 
          (((dx*sqrt(pow(dx,2) + pow(dy,2)))/pow(pow(dx,2) + pow(dy,2) + pow(dz,2),1.5) - 
          dx/(sqrt(pow(dx,2) + pow(dy,2))*sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2))))*
          (1 + sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))/
          (2.*sqrt((1 - sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
          (1 + sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))))/2.;
           
 
 dr(2,2) = (((-((dy*sqrt(pow(dx,2) + pow(dy,2)))/pow(pow(dx,2) + pow(dy,2) + pow(dz,2),1.5)) + 
           dy/(sqrt(pow(dx,2) + pow(dy,2))*sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2))))*
           (1 - sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))/
           (2.*sqrt((1 + sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           (1 - sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))) - 
           (((dy*sqrt(pow(dx,2) + pow(dy,2)))/pow(pow(dx,2) + pow(dy,2) + pow(dz,2),1.5) - 
           dy/(sqrt(pow(dx,2) + pow(dy,2))*sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2))))*
           (1 + sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))/
           (2.*sqrt((1 - sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           (1 + sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))))/2.;
           
	
 dr(2,3) = (-(sqrt(pow(dx,2) + pow(dy,2))*dz*(1 - sqrt(pow(px,2) + pow(py,2))/
           sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))/
           (2.*pow(pow(dx,2) + pow(dy,2) + pow(dz,2),1.5)*
           sqrt((1 + sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           (1 - sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))) - 
           (sqrt(pow(dx,2) + pow(dy,2))*dz*(1 + 
           sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))/
           (2.*pow(pow(dx,2) + pow(dy,2) + pow(dz,2),1.5)*
           sqrt((1 - sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           (1 + sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))))/2.;
           

 
//d/dp_i 
//debug: x->p index xhange in denominator

 dr(2,4) = (((1 + sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           ((px*sqrt(pow(px,2) + pow(py,2)))/pow(pow(px,2) + pow(py,2) + pow(pz,2),1.5) - 
           px/(sqrt(pow(px,2) + pow(py,2))*sqrt(pow(px,2) + pow(py,2) + pow(pz,2)))))/
           (2.*sqrt((1 + sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           (1 - sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))) - 
           ((1 - sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           (-((px*sqrt(pow(px,2) + pow(py,2)))/pow(pow(px,2) + pow(py,2) + pow(pz,2),1.5)) + 
           px/(sqrt(pow(px,2) + pow(py,2))*sqrt(pow(px,2) + pow(py,2) + pow(pz,2)))))/
           (2.*sqrt((1 - sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           (1 + sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))))/2.;
           
 
 dr(2,5) = (((1 + sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           ((py*sqrt(pow(px,2) + pow(py,2)))/pow(pow(px,2) + pow(py,2) + pow(pz,2),1.5) - 
           py/(sqrt(pow(px,2) + pow(py,2))*sqrt(pow(px,2) + pow(py,2) + pow(pz,2)))))/
           (2.*sqrt((1 + sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           (1 - sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))) - 
           ((1 - sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           (-((py*sqrt(pow(px,2) + pow(py,2)))/pow(pow(px,2) + pow(py,2) + pow(pz,2),1.5)) + 
           py/(sqrt(pow(px,2) + pow(py,2))*sqrt(pow(px,2) + pow(py,2) + pow(pz,2)))))/
           (2.*sqrt((1 - sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           (1 + sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))))/2.;
           
 
 dr(2,6) = (((1 + sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           sqrt(pow(px,2) + pow(py,2))*pz)/
           (2.*pow(pow(px,2) + pow(py,2) + pow(pz,2),1.5)*
           sqrt((1 + sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           (1 - sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))) + 
           ((1 - sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
            sqrt(pow(px,2) + pow(py,2))*pz)/
           (2.*pow(pow(px,2) + pow(py,2) + pow(pz,2),1.5)*
           sqrt((1 - sqrt(pow(dx,2) + pow(dy,2))/sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2)))*
           (1 + sqrt(pow(px,2) + pow(py,2))/sqrt(pow(px,2) + pow(py,2) + pow(pz,2))))))/2.;
           
 
 dr(2,7) = 0;
 
// cout<<"derivative matrix "<<dr<<endl;
 return std::pair<AlgebraicMatrix,AlgebraicVector>(dr,point); 
}
