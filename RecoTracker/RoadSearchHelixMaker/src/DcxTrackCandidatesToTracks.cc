// -------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DcxTrackCandidatesToTracks.cc,v 1.7 2007/01/15 22:16:28 gutsche Exp $
//
// Description:
//	Class Implementation for |DcxTrackCandidatesToTracks|
//
// Environment:
//	Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//	S. Wagner
//
// Copyright Information:
//	Copyright (C) 1995	SLAC
//
//------------------------------------------------------------------------
#include <cmath>
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxTrackCandidatesToTracks.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/Dcxmatinv.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxFittedHel.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHit.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/Dcxprobab.hh"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

// #include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryParameters.h"
// #include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"

#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToPointBuilder.h"

#include "DataFormats/Math/interface/Vector3D.h"

using std::cout;
using std::endl;
using std::ostream;

double DcxTrackCandidatesToTracks::epsilon      =   0.000000001;
double DcxTrackCandidatesToTracks::half_pi=1.570796327;

//constructors
DcxTrackCandidatesToTracks::DcxTrackCandidatesToTracks(){
  edm::LogInfo("RoadSearch") << "DcxTrackCandidatesToTracks null constructor - does nothing" ;}

//points
DcxTrackCandidatesToTracks::DcxTrackCandidatesToTracks(std::vector<DcxHit*> &listohits, reco::TrackCollection &output, const MagneticField *field)
{ 
  double rmin=1000.0; double phi_try=0.0; int ntrk=0;
  for (unsigned int i=0; i<listohits.size(); ++i) {
    if (!listohits[i]->stereo()){
      double rhit=sqrt(listohits[i]->x()*listohits[i]->x()+listohits[i]->y()*listohits[i]->y());
      if ( (rhit<rmin) ){phi_try=atan2(listohits[i]->y(),listohits[i]->x());rmin=rhit;}
    }
    for (unsigned int j=0; j<listohits.size(); ++j) {
      for (unsigned int k=0; k<listohits.size(); ++k) {
        if ( ( ( 1==listohits[i]->Layer())||( 3==listohits[i]->Layer()) ) 
	     && ( ( 9==listohits[j]->Layer())||(11==listohits[j]->Layer()) )
	     && ( (17==listohits[k]->Layer())||(19==listohits[k]->Layer()) ) ){
          makecircle(listohits[i]->x(),listohits[i]->y(),listohits[j]->x(),listohits[j]->y(),
		     listohits[k]->x(),listohits[k]->y());
          double xc=xc_cs; double yc=yc_cs; double rc=rc_cs;
          double d0=sqrt(xc*xc+yc*yc)-rc; 
// 	  double dmax=sqrt(xc*xc+yc*yc)+rc; // not used
          double s3=s3_cs;
          if ( (fabs(s3)>0.1) ){
	    double d0h=-s3*d0;
	    double phi0h=atan2(yc,xc)+s3*half_pi;
	    double omegah=-s3/rc;
	    DcxHel make_a_circ(d0h,phi0h,omegah,0.0,0.0);// make_a_hel.print();
	    std::vector<DcxHit*> axlist;
	    check_axial( listohits, axlist, make_a_circ);
	    int n_axial=axlist.size();
	    for (unsigned int l=0; l<listohits.size(); ++l) {
	      for (unsigned int m=0; m<listohits.size(); ++m) {
		if ( ((2==listohits[l]->Layer())||(4==listohits[l]->Layer())) && 
		     ((12==listohits[m]->Layer())||(10==listohits[m]->Layer())) ){
		  double z1=find_z_at_cyl(make_a_circ,listohits[l]); 
		  double z2=find_z_at_cyl(make_a_circ,listohits[m]);
		  double l1=find_l_at_z(z1,make_a_circ,listohits[l]); 
		  double l2=find_l_at_z(z2,make_a_circ,listohits[m]);
		  double tanl=(z1-z2)/(l1-l2); double z0=z1-tanl*l1;
		  DcxHel make_a_hel(d0h,phi0h,omegah,z0,tanl);// make_a_hel.print();
		  std::vector<DcxHit*> outlist = axlist;
		  check_stereo( listohits, outlist, make_a_hel);
		  int n_stereo=outlist.size()-n_axial;
		  if ((n_stereo>2)&&(n_axial>6)){
		    DcxFittedHel real_fit(outlist,make_a_hel);// real_fit.FitPrint();
		    if (real_fit.Prob()>0.001){
		      ntrk++;
		      edm::LogInfo("RoadSearch") << "ntrk xprob pt nax nst d0 phi0 omega z0 tanl " << ntrk << " " << real_fit.Prob() 
						 << " " << real_fit.Pt() << " " << n_axial << " " << n_stereo
						 << " " << real_fit.D0() << " " << real_fit.Phi0() << " " << real_fit.Omega() 
						 << " " << real_fit.Z0() << " " << real_fit.Tanl() ;


		      AlgebraicVector paraVector(5);
		      paraVector[0] = real_fit.Omega();
		      paraVector[1] = half_pi - atan(real_fit.Tanl());
		      paraVector[2] = real_fit.Phi0();
		      paraVector[3] = - real_fit.D0();
		      paraVector[4] = real_fit.Z0();

		      PerigeeTrajectoryParameters params(paraVector);

		      PerigeeTrajectoryError error;

		      GlobalPoint reference(0,0,0);

		      TrajectoryStateClosestToPoint tscp(params, 
							 real_fit.Pt(),
							 error, 
							 reference,
							 field);

		      GlobalPoint v = tscp.theState().position();
		      math::XYZPoint  pos( v.x(), v.y(), v.z() );
		      GlobalVector p = tscp.theState().momentum();
		      math::XYZVector mom( p.x(), p.y(), p.z() );

		      output.push_back(reco::Track(real_fit.Chisq(),
						   outlist.size()-5,
						   pos, 
						   mom, 
						   tscp.charge(), 
						   tscp.theState().curvilinearError()));

		      for (unsigned int n=0; n<outlist.size(); ++n){outlist[n]->SetUsedOnHel(ntrk);}
		    }else{
		    }
		  }
		}
	      }
	    }
          }
        }
      }
    }
  }
}//endof DcxTrackCandidatesToTracks

//destructor
DcxTrackCandidatesToTracks::~DcxTrackCandidatesToTracks( ){ }//endof ~DcxTrackCandidatesToTracks

void DcxTrackCandidatesToTracks::makecircle(double x1_cs, double y1_cs, double x2_cs,
				   double y2_cs, double x3_cs, double y3_cs){
  x1t_cs=x1_cs-x3_cs; y1t_cs=y1_cs-y3_cs; r1s_cs=x1t_cs*x1t_cs+y1t_cs*y1t_cs;
  x2t_cs=x2_cs-x3_cs; y2t_cs=y2_cs-y3_cs; r2s_cs=x2t_cs*x2t_cs+y2t_cs*y2t_cs;
  double rho=x1t_cs*y2t_cs-x2t_cs*y1t_cs;
  if (fabs(rho)<DcxTrackCandidatesToTracks::epsilon){
    rc_cs=1.0/(DcxTrackCandidatesToTracks::epsilon);
    fac_cs=sqrt(x1t_cs*x1t_cs+y1t_cs*y1t_cs);
    xc_cs=x2_cs+y1t_cs*rc_cs/fac_cs;
    yc_cs=y2_cs-x1t_cs*rc_cs/fac_cs;
  }else{
    fac_cs=0.5/rho;
    xc_cs=fac_cs*(r1s_cs*y2t_cs-r2s_cs*y1t_cs);
    yc_cs=fac_cs*(r2s_cs*x1t_cs-r1s_cs*x2t_cs); 
    rc_cs=sqrt(xc_cs*xc_cs+yc_cs*yc_cs); xc_cs+=x3_cs; yc_cs+=y3_cs;
  }
  s3_cs=0.0;
  f1_cs=x1_cs*yc_cs-y1_cs*xc_cs; f2_cs=x2_cs*yc_cs-y2_cs*xc_cs; 
  f3_cs=x3_cs*yc_cs-y3_cs*xc_cs;
  if ((f1_cs<0.0)&&(f2_cs<0.0)&&(f3_cs<0.0))s3_cs=1.0;
  if ((f1_cs>0.0)&&(f2_cs>0.0)&&(f3_cs>0.0))s3_cs=-1.0;
}

void DcxTrackCandidatesToTracks::check_axial( std::vector<DcxHit*> &listohits, std::vector<DcxHit*> &outlist, DcxHel make_a_hel){
  for (unsigned int i=0; i<listohits.size(); ++i) {
    DcxHit* try_me = listohits[i];
    if ((!try_me->stereo())&&(!try_me->GetUsedOnHel())){
      double doca = try_me->residual(make_a_hel); 
//      double len = make_a_hel.Doca_Len();
//      edm::LogInfo("RoadSearch") << "In check_axial, doca(mmm) len xh yh " << 10000.0*doca << " " << len 
// 			      << " " << make_a_hel.Xh(len) << " " << make_a_hel.Yh() ;
      if (fabs(doca)<0.100)outlist.push_back(try_me);
    } 
  }
}

void DcxTrackCandidatesToTracks::check_stereo( std::vector<DcxHit*> &listohits, std::vector<DcxHit*> &outlist, DcxHel make_a_hel){
  for (unsigned int i=0; i<listohits.size(); ++i) {
    DcxHit* try_me = listohits[i];
    if ((try_me->stereo())&&(!try_me->GetUsedOnHel())){
      double doca = try_me->residual(make_a_hel);
//      double len = make_a_hel.Doca_Len();
//      edm::LogInfo("RoadSearch") << "In check_stereo, doca(mmm) len xh yh " << 10000.0*doca << " " << len 
// 			      << " " << make_a_hel.Xh(len) << " " << make_a_hel.Yh(len) ;
      if (fabs(doca)<0.100)outlist.push_back(try_me);
    } 
  }
}

double DcxTrackCandidatesToTracks::find_z_at_cyl(DcxHel he, DcxHit* hi){
  zint=-1000.0;
  x0_wr=hi->x(); y0_wr=hi->y(); sx_wr=hi->wx(); sy_wr=hi->wy();
  xc_cl=he.Xc(); yc_cl=he.Yc(); r_cl=fabs(1.0/he.Omega()); 
  double cx=x0_wr-xc_cl; double cy=y0_wr-yc_cl;
  double a=sx_wr*sx_wr+sy_wr*sy_wr; double b=2.0*(sx_wr*cx+sy_wr*cy); 
  double c=(cx*cx+cy*cy-r_cl*r_cl);
  double bsq=b*b; double fac=4.0*a*c; double ta=2.0*a;
  if (fac < bsq){
    double sfac=sqrt(bsq-fac); double z1=-(b-sfac)/ta; double z2=-(b+sfac)/ta;
    if (fabs(z1) < fabs(z2)){
      zint=z1;
    }else{
      zint=z2;
    }
  }
  return zint;
}
double DcxTrackCandidatesToTracks::find_l_at_z(double zi, DcxHel he, DcxHit* hi){
  double omega=he.Omega(), d0=he.D0();
  double xl = hi->x()+zi*hi->wx()/hi->wz();
  double yl = hi->y()+zi*hi->wy()/hi->wz();
  double rl = sqrt(xl*xl+yl*yl);
  double orcsq=(1.0+d0*omega)*(1.0+d0*omega);    
  double orlsq=(rl*omega)*(rl*omega);
  double cphil=(1.0+orcsq-orlsq)/(2.0*(1.0+d0*omega));
  double phil1=acos(cphil);
  double l1=fabs(phil1/omega); 
  return l1;
}
