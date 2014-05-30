#ifndef L1TDISK_H
#define L1TDISK_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
using namespace std;

#include "L1TStub.hh"
#include "L1TTracklet.hh"
#include "L1TGeomBase.hh"

class L1TDisk: public L1TGeomBase{

private:
  L1TDisk(){
  }


public:

  L1TDisk(double zmin,double zmax, int NSector){
    zmin_=zmin;
    zmax_=zmax;
    if (zmin<0.0) {
      zmin_=zmax;
      zmax_=zmin;
    }
    NSector_=NSector;
    stubs_.resize(NSector);
    tracklets_.resize(NSector);
    tracks_.resize(NSector);
  }

  bool addStub(const L1TStub& aStub){
    //cout << "z zmin zmax:"<<aStub.z()<<" "<<zmin_<<" "<<zmax_<<endl;
    if (aStub.z()<zmin_||aStub.z()>zmax_) return false;
    double phi=aStub.phi();
    if (phi<0) phi+=two_pi;
    int nSector=NSector_*phi/two_pi;
    assert(nSector>=0);
    assert(nSector<NSector_);

    stubs_[nSector].push_back(aStub);
    return true;
  }

  void findTracklets(L1TDisk* D){

    for(int iSector=0;iSector<NSector_;iSector++){
      for (int offset=-1;offset<2;offset++) {
	int jSector=iSector+offset;
	if (jSector<0) jSector+=NSector_;
	if (jSector>=NSector_) jSector-=NSector_;
	for (unsigned int i=0;i<stubs_[iSector].size();i++) {
	  double r1=stubs_[iSector][i].r();
	  double z1=stubs_[iSector][i].z();
	  double phi1=stubs_[iSector][i].phi();
	  for (unsigned int j=0;j<D->stubs_[jSector].size();j++) {
	    double r2=D->stubs_[jSector][j].r();
	    double z2=D->stubs_[jSector][j].z();
	    double zcrude=z1-(z2-z1)*r1/(r2-r1);
	    if (fabs(zcrude)>30) continue;


	    double phi2=D->stubs_[jSector][j].phi();
	    
	    if (r1>60.0||r2>60.0) continue;

	    double deltaphi=phi1-phi2;

	    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	    assert(fabs(deltaphi)<0.5*two_pi);

	    if (fabs(r1-r2)/fabs(z1-z2)<0.1) continue;
	    
	    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
        
	    double rinv=2*sin(deltaphi)/dist;

	    if (fabs(rinv)>0.0057) continue;

	    double tmp=0.5*r1*rinv;
	    
	    if (fabs(tmp)>=1.0) continue;
	    
	    double phi0=phi1+asin(tmp);

	    if (phi0>0.5*two_pi) phi0-=two_pi;
	    if (phi0<-0.5*two_pi) phi0+=two_pi;
	    if (!(fabs(phi0)<0.5*two_pi)) {
	      cout << "phi0 "<<phi0<<" "<<phi1<<" "<<r1<<" "<<rinv<<endl;
	      cout << "r1 r2 deltaphi "<<r1<<" "<<r2<<" "<<deltaphi<<endl;
	      cout << "z1 z2 "<<z1<<" "<<z2<<endl;
	    }
	    assert(fabs(phi0)<0.5*two_pi);

	    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
	    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
	    double t=(z1-z2)/(rhopsi1-rhopsi2);

	    double z0=z1-t*rhopsi1;

	    if (fabs(z0)>15.0) continue;

	    double pt1=stubs_[iSector][i].pt();
	    double pt2=D->stubs_[jSector][j].pt();
	    double pttracklet=0.3*3.8/(rinv*100);
	    bool pass1=fabs(1.0/pt1-1.0/pttracklet)<0.5;
	    bool pass2=fabs(1.0/pt2-1.0/pttracklet)<0.5;
	    bool pass=pass1&&pass2;
	    if (!pass) continue;

	    //cout << "L1TDisk found tracklet"<<endl;
 
	    L1TTracklet tracklet(rinv,phi0,t,z0);
	    tracklet.addStub(stubs_[iSector][i]);
	    tracklet.addStub(D->stubs_[jSector][j]);

	    //cout << "L1TDisk tracklet z() "<<tracklet.z()<<endl;

	    tracklets_[iSector].push_back(tracklet);



	    //cout << "rinv phi0 t z0:"<<
	    //  rinv<<" "<<phi0<<" "<<t<<" "<<z0<<endl;

	  }
	}
      }
    }
  }

  void findMatches(L1TDisk* D, double phiSF, double rphicut1, double rcut1, 
		   double rphicut2=0.2, double rcut2=3.0){


    for(int iSector=0;iSector<NSector_;iSector++){
      for (unsigned int i=0;i<tracklets_[iSector].size();i++) {
	L1TTracklet& aTracklet=tracklets_[iSector][i];
	double rinv=aTracklet.rinv();
	double phi0=aTracklet.phi0();
	double z0=aTracklet.z0();
	double t=aTracklet.t();

	double bestdist=2e30;
	L1TStub tmp;

	for (int offset=-1;offset<2;offset++) {
	  int jSector=iSector+offset;
	  if (jSector<0) jSector+=NSector_;
	  if (jSector>=NSector_) jSector-=NSector_;
	  if (D->stubs_[jSector].size()==0) continue;

	  double zapprox=D->stubs_[jSector][0].z();

	  double r_track_approx=2.0*sin(0.5*rinv*(zapprox-z0)/t)/rinv;
	  double phi_track_approx=phi0-0.5*rinv*(zapprox-z0)/t;
	  if (phi_track_approx-D->stubs_[jSector][0].phi()<-0.5*two_pi) phi_track_approx+=two_pi;  
	  if (phi_track_approx-D->stubs_[jSector][0].phi()>0.5*two_pi) phi_track_approx-=two_pi;  

	  
	  for (unsigned int j=0;j<D->stubs_[jSector].size();j++) {
	    double r=D->stubs_[jSector][j].r();
	    if (fabs(r-r_track_approx)>10.0) continue;
	    double z=D->stubs_[jSector][j].z();
	    double phi=D->stubs_[jSector][j].phi();

	    if (fabs((phi-phi_track_approx)*r_track_approx)>1.0) continue;
	    
	    double r_track=2.0*sin(0.5*rinv*(z-z0)/t)/rinv;
	    double phi_track=phi0-0.5*rinv*(z-z0)/t;

	    int iphi=D->stubs_[jSector][j].iphi();
	    double width=4.608;
	    double nstrip=508.0;
	    if (r<60.0) {
	      width=4.8;
	      nstrip=480;
	    }
	    double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...
	    if (z>0.0) Deltai=-Deltai;
	    

	    double theta0=asin(Deltai/r);

	    double Delta=Deltai-r_track*sin(theta0-(phi_track-phi));



	    if (Delta!=Delta) {
	      cout << "Error: "<<t<<" "<<rinv<<" "<<theta0<<endl;
	      continue;
	    }
	    
	    double phiproj=phi0-0.5*(z-z0)*rinv/t;
	    double rproj=2.0*sin(0.5*(z-z0)*rinv/t)/rinv;

	    double deltaphi=phi-phiproj;
	    //cout << "deltaphi phi phiproj:"<<deltaphi<<" "<<phi<<" "<<phiproj<<" "<<phi0<<" "<<asin(0.5*r*rinv)<<endl;

	    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	    assert(fabs(deltaphi)<0.5*two_pi);

	    double rdeltaphi=Delta;
            double deltar=r-rproj;

	    if (0&&fabs(Delta)<10.0&&fabs(deltar)<15.0) {
	      static ofstream out("diskmatch.txt");
	      out << aTracklet.r()<<" "<<aTracklet.z()<<" "<<r<<" "<<z<<" "
		  <<Delta<<" "<<deltar<<endl;
	    }

	    double dist=0.0;

	    if (r<60) {
	      if (fabs(rdeltaphi)>rphicut1*phiSF) continue;
	      if (fabs(deltar)>rcut1) continue;
	      dist=hypot(rdeltaphi/rphicut1,deltar/rcut1);
	    }
	    else {
	      if (fabs(rdeltaphi)>rphicut2*phiSF) continue;
	      if (fabs(deltar)>rcut2) continue;
	      dist=hypot(rdeltaphi/rphicut2,deltar/rcut2);
	    }

	    /*
	    double pt1=D->stubs_[jSector][j].pt();
	    double pttracklet=aTracklet.pt(3.8);
	    bool pass1=fabs(1.0/pt1-1.0/pttracklet)<0.5;
	    if (!pass1) continue;
	    */

	    if (dist<bestdist){
	      bestdist=dist;
	      tmp=D->stubs_[jSector][j];
	    }

	    //cout << "rdeltaphi deltar:"<<rdeltaphi<<" "<<deltar<<endl;

	  }
	}
	
	if (bestdist<1e30) {
	  tracklets_[iSector][i].addStub(tmp);
	}
      }
    }
  }


  void findBarrelMatches(L1TGeomBase* L, double phiSF){

    for(int iSector=0;iSector<NSector_;iSector++){
      for (unsigned int i=0;i<tracklets_[iSector].size();i++) {
	L1TTracklet& aTracklet=tracklets_[iSector][i];
	double rinv=aTracklet.rinv();
	double phi0=aTracklet.phi0();
	double z0=aTracklet.z0();
	double t=aTracklet.t();

	double bestdist=2e30;
	L1TStub tmp;

	for (int offset=-1;offset<2;offset++) {
	  int jSector=iSector+offset;
	  if (jSector<0) jSector+=NSector_;
	  if (jSector>=NSector_) jSector-=NSector_;
	  if (L->stubs_[jSector].size()==0) continue;
	  

	  double rapprox=L->stubs_[jSector][0].r();

	  double phiprojapprox=phi0-asin(0.5*rapprox*rinv);
	  double zprojapprox=z0+2*t*asin(0.5*rapprox*rinv)/rinv;
	  if (phiprojapprox-L->stubs_[jSector][0].phi()<-0.5*two_pi) phiprojapprox+=two_pi;  
	  if (phiprojapprox-L->stubs_[jSector][0].phi()>0.5*two_pi) phiprojapprox-=two_pi;  
	  

	  for (unsigned int j=0;j<L->stubs_[jSector].size();j++) {
	    double r=L->stubs_[jSector][j].r();
	    double z=L->stubs_[jSector][j].z();
	    if (fabs(z-zprojapprox)>15.0) continue;
	    double phi=L->stubs_[jSector][j].phi();
	    double deltaphiapprox=fabs(phi-phiprojapprox);
	    assert(deltaphiapprox<12.0);
	    if (deltaphiapprox*rapprox>2.0) continue;
	    //cout << "r1 phi1 r2 phi2:"
	    //  <<stubs_[iSector][i].r()<<" "
	    //  <<stubs_[iSector][i].phi()<<" "
	    //  <<L->stubs_[jSector][j].r()<<" "
	    //  <<L->stubs_[jSector][j].phi()<<endl;

	    
	    double phiproj=phi0-asin(0.5*r*rinv);
	    double zproj=z0+2*t*asin(0.5*r*rinv)/rinv;

	    double deltaphi=phi-phiproj;
	    //cout << "deltaphi phi phiproj:"<<deltaphi<<" "<<phi<<" "<<phiproj<<" "<<phi0<<" "<<asin(0.5*r*rinv)<<endl;

	    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	    assert(fabs(deltaphi)<0.5*two_pi);

	    double rdeltaphi=r*deltaphi;
            double deltaz=z-zproj;

	    if (fabs(rdeltaphi)>0.1*phiSF) continue;
	    if (fabs(deltaz)>5.0) continue; //LS modified from 0.5 to 5.0

	    double dist=hypot(rdeltaphi/0.1,deltaz/5.0); //LS modified from 0.5 to 5.0

	    if (dist<bestdist) {
	      bestdist=dist;
	      tmp=L->stubs_[jSector][j];
	    }

	  }
	}
	if (bestdist<1e30) {
	  tracklets_[iSector][i].addStub(tmp);
	}
      }
    }
  }



private:

  double zmin_;
  double zmax_;

};



#endif



