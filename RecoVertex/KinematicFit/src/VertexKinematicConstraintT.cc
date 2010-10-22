#include "RecoVertex/KinematicFit/interface/VertexKinematicConstraintT.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

VertexKinematicConstraintT::VertexKinematicConstraintT()
{}

VertexKinematicConstraintT::~VertexKinematicConstraintT()
{}


void VertexKinematicConstraintT::init(const std::vector<KinematicState>& states,
				      const GlobalPoint& ipoint,  const GlobalVector& fieldValue) {
  int num = states.size();
  if(num!=2) throw VertexException("VertexKinematicConstraintT !=2 states passed");
  point = ipoint;
  mfz = fieldValue.z();
  
  int j=0;
  for(std::vector<KinematicState>::const_iterator i = states.begin(); i != states.end(); i++)
    {
      ch[j] = i->particleCharge();
      mom[j] = i->globalMomentum();
      pos[j] =  i->globalPosition();
      ++j;
    }
}



ROOT::Math::SVector<double,4> VertexKinematicConstraintT::value() const
{
  //it is 2 equations per track
  ROOT::Math::SVector<double,4> vl;
  for(int num_r = 0; num_r!=2; ++num_r) {
    double d_x = point.x() - pos[num_r].x();
    double d_y = point.y() - pos[num_r].y();
    double d_z = point.z() - pos[num_r].z();
    double pt = mom[num_r].transverse();
    if(ch[num_r] !=0) {
      
      //charged particle
      double a_i = - ch[num_r]*mfz;
      double j = a_i*(d_x * mom[num_r].x() + d_y * mom[num_r].y())/(pt*pt);
      if(fabs(j)>1.0){
	LogDebug("VertexKinematicConstraint")
	  << "Warning! asin("<<j<<")="<<asin(j)<<". Fit will be aborted.\n";
      }
      
      
      //vector of values
      vl(num_r*2) = d_y*mom[num_r].x() - d_x*mom[num_r].y() -a_i*(d_x*d_x + d_y*d_y)/2;
      vl(num_r*2 +1) = d_z - mom[num_r].z()*asin(j)/a_i;
    }else{      
      //neutral particle
      vl(num_r*2) = d_y*mom[num_r].x() - d_x*mom[num_r].y();
      vl(num_r*2 +1) = d_z - mom[num_r].z()*(d_x * mom[num_r].x() + d_y * mom[num_r].y())/(pt*pt);
    }
  }
  return vl;
}

ROOT::Math::SMatrix<double, 4,14> VertexKinematicConstraintT::parametersDerivative() const
{
  ROOT::Math::SMatrix<double,4,14> jac_d;
  ROOT::Math::SMatrix<double,2,7> el_part_d;
  for(int num_r = 0; num_r!=2; ++num_r) {
    double d_x = point.x() - pos[num_r].x();
    double d_y = point.y() - pos[num_r].y();
    // double d_z = point.z() - pos[num_r].z();
    double pt2Inverse = 1./mom[num_r].perp2();
    
    if(ch[num_r] !=0) {
      
      //charged particle
      double a_i = - ch[num_r] * mfz;
      double j = a_i*(d_x * mom[num_r].x() + d_y * mom[num_r].y())*pt2Inverse;
      double r_x = d_x - 2* mom[num_r].x()*(d_x*mom[num_r].x()+d_y*mom[num_r].y())*pt2Inverse;
      double r_y = d_y - 2* mom[num_r].y()*(d_x*mom[num_r].x()+d_y*mom[num_r].y())*pt2Inverse;
      double s =  pt2Inverse/sqrt(1 - j*j);
       
      if(fabs(j)>1.0){
	LogDebug("VertexKinematicConstraint")
	  << "Warning! asin("<<j<<")="<<asin(j)<<". Fit will be aborted.\n";
      }
      
      //D Jacobian matrix
      el_part_d(0,0) =  mom[num_r].y() + a_i*d_x;
      el_part_d(0,1) = -mom[num_r].x() + a_i*d_y;
      el_part_d(1,0) =  mom[num_r].x() * (mom[num_r].z() * s);
      el_part_d(1,1) =  mom[num_r].y() * (mom[num_r].z() * s);
      el_part_d(1,2) = -1.;
      el_part_d(0,3) = d_y;
      el_part_d(0,4) = -d_x;
      el_part_d(1,3) = -mom[num_r].z()*s*r_x;
      el_part_d(1,4) = -mom[num_r].z()*s*r_y;
      el_part_d(1,5) = -asin(j) /a_i;
      jac_d.Place_at(el_part_d,num_r*2, num_r*7);
    }else{
      //neutral particle
      el_part_d(0,0) =  mom[num_r].y();
      el_part_d(0,1) = -mom[num_r].x();
      el_part_d(1,0) =  mom[num_r].x() * (mom[num_r].z()*pt2Inverse);
      el_part_d(1,1) =  mom[num_r].y() * (mom[num_r].z()*pt2Inverse);
      el_part_d(1,2) = -1.;
      el_part_d(0,3) = d_y;
      el_part_d(0,4) = -d_x;
      el_part_d(1,3) = 2*(d_x*mom[num_r].x()+d_y*mom[num_r].y())*pt2Inverse*mom[num_r].x()*(mom[num_r].z()*pt2Inverse) - d_x*(mom[num_r].z()*pt2Inverse);
      el_part_d(1,4) = 2*(d_x*mom[num_r].x()+d_y*mom[num_r].y())*pt2Inverse*mom[num_r].y()*(mom[num_r].z()*pt2Inverse) - d_x*(mom[num_r].z()*pt2Inverse);
      el_part_d(1,5) = - (d_x*mom[num_r].x()+d_y*mom[num_r].y())*pt2Inverse;
      jac_d.Place_at(el_part_d,num_r*2, num_r*7);
    }
  }
  return jac_d;
}

ROOT::Math::SMatrix<double, 4,3> VertexKinematicConstraintT::positionDerivative() const
{
  ROOT::Math::SMatrix<double,4,3> jac_e;
  ROOT::Math::SMatrix<double,2,3>  el_part_e;
  for(int num_r = 0; num_r!=2; ++num_r) {
    double d_x = point.x() - pos[num_r].x();
    double d_y = point.y() - pos[num_r].y();
    // double d_z = point.z() - pos[num_r].z();
    double pt2Inverse = 1./mom[num_r].perp2();

    if(ch[num_r] !=0) {
      
      //charged particle
      double a_i = - ch[num_r] * mfz;
      double j = a_i*(d_x * mom[num_r].x() + d_y * mom[num_r].y())*pt2Inverse;
      double s =  pt2Inverse/sqrt(1 - j*j);
      
      //E jacobian matrix
      el_part_e(0,0) = -(mom[num_r].y() + a_i*d_x);
      el_part_e(0,1) = mom[num_r].x() - a_i*d_y;
      el_part_e(1,0) = -mom[num_r].x()*mom[num_r].z()*s;
      el_part_e(1,1) = -mom[num_r].y()*mom[num_r].z()*s;
      el_part_e(1,2) = 1;
      jac_e.Place_at(el_part_e,2*num_r,0);
    }else{      
      //neutral particle
      el_part_e(0,0) = - mom[num_r].y();
      el_part_e(0,1) = mom[num_r].x();
      el_part_e(1,0) = -mom[num_r].x()*mom[num_r].z()*pt2Inverse;
      el_part_e(1,1) = -mom[num_r].y()*mom[num_r].z()*pt2Inverse;
      el_part_e(1,2) = 1;
      jac_e.Place_at(el_part_e,2*num_r,0);
    }
  }
  return jac_e;
}

int VertexKinematicConstraintT::numberOfEquations() const
{return 2;}
