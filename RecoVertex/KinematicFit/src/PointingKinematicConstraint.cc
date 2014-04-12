#include "RecoVertex/KinematicFit/interface/PointingKinematicConstraint.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

std::pair<AlgebraicVector, AlgebraicVector> PointingKinematicConstraint::value(const AlgebraicVector& exPoint) const
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

std::pair<AlgebraicMatrix, AlgebraicVector> PointingKinematicConstraint::derivative(const AlgebraicVector& exPoint) const
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

std::pair<AlgebraicMatrix, AlgebraicVector> PointingKinematicConstraint::derivative(const std::vector<RefCountedKinematicParticle> &par) const
{
 int nStates = par.size();
 if(nStates == 0) throw VertexException("PointingKinematicConstraint::Empty vector of particles passed");
 if(nStates != 1) throw VertexException("PointingKinematicConstraint::Current version does not support the multistate refit");
 
 AlgebraicMatrix dr(2,7,0);
 AlgebraicVector lPoint = asHepVector<7>(par.front()->currentState().kinematicParameters().vector());

//2x7 derivative matrix for given state  
 AlgebraicMatrix lDeriv = makeDerivative(lPoint).first;
 dr.sub(1,1,lDeriv);
 return std::pair<AlgebraicMatrix,AlgebraicVector>(dr,lPoint);
}

std::pair<AlgebraicVector, AlgebraicVector> PointingKinematicConstraint::value(const std::vector<RefCountedKinematicParticle> &par) const
{
 int nStates = par.size();
 if(nStates == 0) throw VertexException("PointingKinematicConstraint::Empty vector of particles passed");
 if(nStates != 1) throw VertexException("PointingKinematicConstraint::Current version does not support the multistate refit");
 AlgebraicVector vl(2,0);
 AlgebraicVector lPoint = asHepVector<7>(par.front()->currentState().kinematicParameters().vector());
 vl(1) = makeValue(lPoint).first(1);
 vl(2) = makeValue(lPoint).first(2);
 return std::pair<AlgebraicVector,AlgebraicVector>(vl,lPoint); 
}

std::pair<AlgebraicVector,AlgebraicVector> PointingKinematicConstraint::makeValue(const AlgebraicVector& exPoint)const
{
 AlgebraicVector vl(2,0);
 AlgebraicVector point = exPoint;
 double dx = point(1) - refPoint.x();
 double dy = point(2) - refPoint.y();
 double dz = point(3) - refPoint.z();
 double px = point(4);
 double py = point(5); 
 double pz = point(6);

// tangent solution
// vl(1) = dy/dx - py/px;
// vl(2) = dz/sqrt(dx*dx + dy*dy) - pz/sqrt(px*px + py*py); 


//half angle solution
 double sin_p = py/sqrt(px*px + py*py);
 double cos_p = px/sqrt(px*px + py*py);
 double sin_x = dy/sqrt(dx*dx + dy*dy);
 double cos_x = dx/sqrt(dx*dx + dy*dy);
 
 double sin_pt = pz/sqrt(px*px + py*py + pz*pz); 
 double cos_pt = sqrt(px*px + py*py)/sqrt(px*px + py*py + pz*pz);
 double sin_xt = dz/sqrt(dx*dx + dy*dy + dz*dz); 
 double cos_xt = sqrt(dx*dx + dy*dy)/sqrt(dx*dx + dy*dy + dz*dz);
 
 vl(1) = (1-cos_x)/sin_x - (1-cos_p)/sin_p;
 vl(2) = (1-cos_xt)/sin_xt - (1-cos_pt)/sin_pt;

//half angle corrected
// vl(1) = (sin_x/(1+cos_x)) - (sin_p/(1+cos_p));
// vl(2) = (sin_xt/(1+cos_xt)) - (sin_pt/(1+cos_pt));
 return std::pair<AlgebraicVector,AlgebraicVector>(vl,point);
}


std::pair<AlgebraicMatrix, AlgebraicVector> PointingKinematicConstraint:: makeDerivative(const AlgebraicVector& exPoint) const
{
 AlgebraicMatrix dr(2,7,0);
 AlgebraicVector point = exPoint;
 double dx = point(1) - refPoint.x();
 double dy = point(2) - refPoint.y();
 double dz = point(3) - refPoint.z();
 double px = point(4);
 double py = point(5); 
 double pz = point(6);
                  
// double tr = px*px + py*py;
// double trd = dx*dx + dy*dy;
// double pr =1.5;
// double p_factor = pow(tr,pr);
// double x_factor = pow(trd,pr);
  
//tangent solution
/*
 dr(1,1) = -dy/(dx*dx);
 dr(1,2) = 1/dx;
 dr(1,3) = 0;
 dr(1,4) = py/(px*px);
 dr(1,5) = -1/px;
 dr(1,6) = 0;
 dr(1,7) = 0;
 
 dr(2,1) = -(dx*dz)/x_factor;
 dr(2,2) = -(dy*dz)/x_factor;
 dr(2,3) = 1/sqrt(dx*dx + dy*dy);
 dr(2,4) = (px*pz)/p_factor;
 dr(2,5) = (py*pz)/p_factor;
 dr(2,6) = -1/sqrt(px*px + py*py);
 dr(2,7) = 0.;
*/
//half angle solution corrected   
/*
 dr(1,1) = - dy/(dx*dx+dy*dy+dx*sqrt(dx*dx+dy*dy));
 dr(1,2) =   dx/(dx*dx+dy*dy+dx*sqrt(dx*dx+dy*dy));
 dr(1,3) = 0; 
 dr(1,4) = py/(px*px+py*py+px*sqrt(px*px+py*py));
 dr(1,5) = -px/(px*px+py*py+px*sqrt(px*px+py*py));
 dr(1,6) = 0;
 dr(1,7) = 0; 
*/

//half angle solution
 dr(1,1) = dx/(dy*sqrt(dx*dx + dy*dy)) - 1/dy;
 dr(1,2) = 1/sqrt(dx*dx+dy*dy) - sqrt(dx*dx+dy*dy)/(dy*dy)+ dx/(dy*dy);
 dr(1,3) = 0; 
 dr(1,4) = -(px/(py*sqrt(px*px + py*py)) - 1/py);
 dr(1,5) = -(1/sqrt(px*px+py*py) - sqrt(px*px+py*py)/(py*py)+ px/(py*py));
 dr(1,6) = 0;
 dr(1,7) = 0; 


//half angle solution
 dr(2,1) = (dx/dz)*(1/sqrt(dx*dx + dy*dy + dz*dz) - 1/sqrt(dx*dx + dy*dy));
 dr(2,2) = (dy/dz)*(1/sqrt(dx*dx + dy*dy + dz*dz) - 1/sqrt(dx*dx + dy*dy));
 dr(2,3) = (1/(dz*dz))*(sqrt(dx*dx + dy*dy) - sqrt(dx*dx+dy*dy+dz*dz)) + 1/sqrt(dx*dx+dy*dy+dz*dz);
 dr(2,4) = -(px/pz)*(1/sqrt(px*px + py*py + pz*pz) - 1/sqrt(px*px + py*py));
 dr(2,5) = -(py/pz)*(1/sqrt(px*px + py*py + pz*pz) - 1/sqrt(px*px + py*py));
 dr(2,6) = -((1/(pz*pz))*(sqrt(px*px + py*py) - sqrt(px*px+py*py+pz*pz)) + 1/sqrt(px*px+py*py+pz*pz));
 dr(2,7) = 0;
 
 return std::pair<AlgebraicMatrix,AlgebraicVector>(dr,point);
}

AlgebraicVector PointingKinematicConstraint::deviations(int nStates) const
{return AlgebraicVector(7*nStates,0);}

int PointingKinematicConstraint::numberOfEquations() const
{return 2;}
