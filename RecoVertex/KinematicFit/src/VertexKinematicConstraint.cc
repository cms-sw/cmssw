#include "RecoVertex/KinematicFit/interface/VertexKinematicConstraint.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

VertexKinematicConstraint::VertexKinematicConstraint()
{}

VertexKinematicConstraint::~VertexKinematicConstraint()
{}

AlgebraicVector VertexKinematicConstraint::value(const std::vector<KinematicState> &states,
                        const GlobalPoint& point) const
{
 int num = states.size();
 if(num<2) throw VertexException("VertexKinematicConstraint::<2 states passed");

//it is 2 equations per track
 AlgebraicVector  vl(2*num,0);
 int num_r = 0;
 for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++)
 {
  TrackCharge ch = i->particleCharge();
  GlobalVector mom = i->globalMomentum();
  GlobalPoint pos =  i->globalPosition();
  double d_x = point.x() - pos.x();
  double d_y = point.y() - pos.y();
  double d_z = point.z() - pos.z();
  double pt = mom.transverse();
  
  if(ch !=0)
  {

//charged particle
   double a_i = - ch * i->magneticField()->inInverseGeV(pos).z();

   double pvx = mom.x() - a_i*d_y;
   double pvy = mom.y() + a_i*d_x;
   double n = a_i*(d_x * mom.x() + d_y * mom.y());
   double m = (pvx*mom.x() + pvy*mom.y());
   double delta = atan2(n,m);

//vector of values
   vl(num_r*2 +1) = d_y*mom.x() - d_x*mom.y() -a_i*(d_x*d_x + d_y*d_y)/2;
   vl(num_r*2 +2) = d_z - mom.z()*delta/a_i;
  }else{

//neutral particle
   vl(num_r*2 +1) = d_y*mom.x() - d_x*mom.y();
   vl(num_r*2 +2) = d_z - mom.z()*(d_x * mom.x() + d_y * mom.y())/(pt*pt);
  }
  num_r++;
 }
 return vl;
}

AlgebraicMatrix VertexKinematicConstraint::parametersDerivative(const std::vector<KinematicState> &states,
                        const GlobalPoint& point) const
{
  int num = states.size();
  if(num<2) throw VertexException("VertexKinematicConstraint::<2 states passed");
  AlgebraicMatrix jac_d(2*num,7*num);
  int num_r = 0;
  for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++)
  {
    AlgebraicMatrix el_part_d(2,7,0);
    TrackCharge ch = i->particleCharge();
    GlobalVector mom = i->globalMomentum();
    GlobalPoint pos =  i->globalPosition();
    double d_x = point.x() - pos.x();
    double d_y = point.y() - pos.y();
    double pt = mom.transverse();
    
    if(ch !=0){

  //charged particle
   double a_i = - ch * i->magneticField()->inInverseGeV(pos).z();
   
   double pvx = mom.x() - a_i*d_y;
   double pvy = mom.y() + a_i*d_x;
   double pvt = sqrt(pvx*pvx+pvy*pvy);
   double novera = (d_x * mom.x() + d_y * mom.y());
   double n = a_i*novera;
   double m = (pvx*mom.x() + pvy*mom.y());
   double k = -mom.z()/(pvt*pvt*pt*pt);
   double delta = atan2(n,m);

  //D Jacobian matrix
     el_part_d(1,1) =  mom.y() + a_i*d_x;
     el_part_d(1,2) = -mom.x() + a_i*d_y;
     el_part_d(2,1) =  -k*(m*mom.x() - n*mom.y());
     el_part_d(2,2) =  -k*(m*mom.y() + n*mom.x());
     el_part_d(2,3) = -1.;
     el_part_d(1,4) = d_y;
     el_part_d(1,5) = -d_x;
     el_part_d(2,4) = k*(m*d_x - novera*(2*mom.x() - a_i*d_y));
     el_part_d(2,5) = k*(m*d_y - novera*(2*mom.y() + a_i*d_x));
     el_part_d(2,6) = -delta /a_i;
     jac_d.sub(num_r*2+1, num_r*7+1, el_part_d);
    }else{
  //neutral particle
      el_part_d(1,1) =  mom.y();
      el_part_d(1,2) = -mom.x();
      el_part_d(2,1) =  mom.x() * mom.z()/(pt*pt);
      el_part_d(2,2) =  mom.y() * mom.z()/(pt*pt);
      el_part_d(2,3) = -1.;
      el_part_d(1,4) = d_y;
      el_part_d(1,5) = -d_x;
      el_part_d(2,4) = 2*(d_x*mom.x()+d_y*mom.y())*mom.x()*mom.z()/(pt*pt*pt*pt) - mom.z()*d_x/(pt*pt);
      el_part_d(2,5) = 2*(d_x*mom.x()+d_y*mom.y())*mom.y()*mom.z()/(pt*pt*pt*pt) - mom.z()*d_y/(pt*pt);
      el_part_d(2,6) =-(d_x * mom.x() + d_y * mom.y())/(pt*pt);
      jac_d.sub(num_r*2+1, num_r*7+1, el_part_d);
    }
    num_r++;
  }
  return jac_d;
}

AlgebraicMatrix VertexKinematicConstraint::positionDerivative(const std::vector<KinematicState> &states,
                                    const GlobalPoint& point) const
{
 int num = states.size();
 if(num<2) throw VertexException("VertexKinematicConstraint::<2 states passed");
 AlgebraicMatrix jac_e(2*num,3);
 int num_r = 0;
 for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++)
 {
  AlgebraicMatrix el_part_e(2,3,0);
  TrackCharge ch = i->particleCharge();
  GlobalVector mom = i->globalMomentum();
  GlobalPoint pos =  i->globalPosition();
  double d_x = point.x() - pos.x();
  double d_y = point.y() - pos.y();
  double pt = mom.transverse();
  
  if(ch !=0 )
  {

//charged particle
   double a_i = - ch * i->magneticField()->inInverseGeV(pos).z();

   double pvx = mom.x() - a_i*d_y;
   double pvy = mom.y() + a_i*d_x;
   double pvt = sqrt(pvx*pvx+pvy*pvy);
   double n = a_i*(d_x * mom.x() + d_y * mom.y());
   double m = (pvx*mom.x() + pvy*mom.y());
   double k = -mom.z()/(pvt*pvt*pt*pt);

//E jacobian matrix
   el_part_e(1,1) = -(mom.y() + a_i*d_x);
   el_part_e(1,2) = mom.x() - a_i*d_y;
   el_part_e(2,1) = k*(m*mom.x() - n*mom.y());
   el_part_e(2,2) = k*(m*mom.y() + n*mom.x());
   el_part_e(2,3) = 1;
   jac_e.sub(2*num_r+1,1,el_part_e);
  }else{

//neutral particle
   el_part_e(1,1) = - mom.y();
   el_part_e(1,2) = mom.x();
   el_part_e(2,1) = -mom.x()*mom.z()/(pt*pt);
   el_part_e(2,2) = -mom.y()*mom.z()/(pt*pt);
   el_part_e(2,3) = 1;
   jac_e.sub(2*num_r+1,1,el_part_e);
  }
  num_r++;
 }
 return jac_e;
}

int VertexKinematicConstraint::numberOfEquations() const
{return 2;}
