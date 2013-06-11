#ifndef L1TBARREL_H
#define L1TBARREL_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
using namespace std;

#include "L1TStub.hh"
#include "L1TTracklet.hh"
#include "L1TTrack.hh"
#include "L1TTracks.hh"
#include "L1TGeomBase.hh"
#include "L1TDisk.hh"

class L1TBarrel:public L1TGeomBase {

private:
  L1TBarrel(){
  }


public:

  L1TBarrel(double rmin,double rmax,double zmax, int NSector){
    rmin_=rmin;
    rmax_=rmax;
    zmax_=zmax;
    NSector_=NSector;
    stubs_.resize(NSector);
    tracklets_.resize(NSector);
    tracks_.resize(NSector);
  }

  bool addStub(const L1TStub& aStub){
    if (aStub.r()<rmin_||aStub.r()>rmax_||fabs(aStub.z())>zmax_) return false;
    double phi=aStub.phi();
    if (phi<0) phi+=two_pi;
    int nSector=NSector_*phi/two_pi;
    assert(nSector>=0);
    assert(nSector<NSector_);

    stubs_[nSector].push_back(aStub);
    return true;
  }

  void findTracklets(L1TBarrel* L){

    for(int iSector=0;iSector<NSector_;iSector++){
      for (int offset=-1;offset<2;offset++) {
	int jSector=iSector+offset;
	if (jSector<0) jSector+=NSector_;
	if (jSector>=NSector_) jSector-=NSector_;
	for (unsigned int i=0;i<stubs_[iSector].size();i++) {
	  for (unsigned int j=0;j<L->stubs_[jSector].size();j++) {
	    //cout << "r1 phi1 r2 phi2:"
	    //  <<stubs_[iSector][i].r()<<" "
	    //  <<stubs_[iSector][i].phi()<<" "
	    //  <<L->stubs_[jSector][j].r()<<" "
	    //  <<L->stubs_[jSector][j].phi()<<endl;
	    double r1=stubs_[iSector][i].r();
	    double z1=stubs_[iSector][i].z();
	    double phi1=stubs_[iSector][i].phi();

	    double r2=L->stubs_[jSector][j].r();
	    double z2=L->stubs_[jSector][j].z();
	    double phi2=L->stubs_[jSector][j].phi();
	    
	    double deltaphi=phi1-phi2;

	    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	    assert(fabs(deltaphi)<0.5*two_pi);


	    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
        
	    double rinv=2*sin(deltaphi)/dist;
	    
	    double phi0=phi1+asin(0.5*r1*rinv);

	    if (phi0>0.5*two_pi) phi0-=two_pi;
	    if (phi0<-0.5*two_pi) phi0+=two_pi;
	    assert(fabs(phi0)<0.5*two_pi);

	    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
	    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
	    double t=(z1-z2)/(rhopsi1-rhopsi2);

	    double z0=z1-t*rhopsi1;

	    if (stubs_[iSector][i].sigmaz()>1.0) {
	      if (fabs(z1-z2)<10.0){
		z0=0.0;
		t=z1/rhopsi1;
	      }
	    }

	    if (fabs(z0)>30.0) continue;
	    if (fabs(rinv)>0.0057) continue;

	    L1TTracklet tracklet(rinv,phi0,t,z0);
	    tracklet.addStub(stubs_[iSector][i]);
	    tracklet.addStub(L->stubs_[jSector][j]);

	    tracklets_[iSector].push_back(tracklet);

	    //cout << "rinv phi0 t z0:"<<
	    //  rinv<<" "<<phi0<<" "<<t<<" "<<z0<<endl;

	  }
	}
      }
    }

  }

  void findMatches(L1TBarrel* L,double cutrphi, double cutrz){

    for(int iSector=0;iSector<NSector_;iSector++){
      for (int offset=-1;offset<2;offset++) {
	int jSector=iSector+offset;
	if (jSector<0) jSector+=NSector_;
	if (jSector>=NSector_) jSector-=NSector_;
	if (L->stubs_[jSector].size()==0) continue;
	for (unsigned int i=0;i<tracklets_[iSector].size();i++) {
	  L1TTracklet& aTracklet=tracklets_[iSector][i];
	  double rinv=aTracklet.rinv();
	  double phi0=aTracklet.phi0();
	  double z0=aTracklet.z0();
	  double t=aTracklet.t();

	  int jbest=-1;
	  double distbest=1e30;

	  double rapprox=L->stubs_[jSector][0].r();

	  double phiprojapprox=phi0-asin(0.5*rapprox*rinv);
	  double zprojapprox=z0+2*t*asin(0.5*rapprox*rinv)/rinv;
	  if (phiprojapprox-L->stubs_[jSector][0].phi()<-0.5*two_pi) phiprojapprox+=two_pi;  
	  if (phiprojapprox-L->stubs_[jSector][0].phi()>0.5*two_pi) phiprojapprox-=two_pi;  

	  for (unsigned int j=0;j<L->stubs_[jSector].size();j++) {
	    double z=L->stubs_[jSector][j].z();
	    if (fabs(z-zprojapprox)>10.0) continue;
	    double phi=L->stubs_[jSector][j].phi();
	    double deltaphiapprox=fabs(phi-phiprojapprox);
	    assert(deltaphiapprox<1.0);
	    if (deltaphiapprox*rapprox>5.0) continue;
	    double r=L->stubs_[jSector][j].r();
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

	    if (0) {
	      static ofstream out("barrelmatch.txt");
	      out << aTracklet.r()<<" "<<r<<" "<<rdeltaphi<<" "<<deltaz
		       <<endl;
	    }
	    if (fabs(rdeltaphi)>cutrphi) continue;
	    if (fabs(deltaz)>cutrz) continue;

	    double dist=hypot(rdeltaphi/cutrphi,deltaz/cutrz);

	    if (dist<distbest){
	      jbest=j;
	      distbest=dist;
	    }

	    //cout << "rdeltaphi deltaz:"<<rdeltaphi<<" "<<deltaz<<endl;
	    
	  }
	  if (jbest!=-1) {
	    tracklets_[iSector][i].addStub(L->stubs_[jSector][jbest]);
	  }
	}
      }
    }
  }


  void findMatches(L1TDisk* D){

    for(int iSector=0;iSector<NSector_;iSector++){
      for (int offset=-1;offset<2;offset++) {
	int jSector=iSector+offset;
	if (jSector<0) jSector+=NSector_;
	if (jSector>=NSector_) jSector-=NSector_;
	for (unsigned int i=0;i<tracklets_[iSector].size();i++) {
	  L1TTracklet& aTracklet=tracklets_[iSector][i];
	  double rinv=aTracklet.rinv();
	  double phi0=aTracklet.phi0();
	  double z0=aTracklet.z0();
	  double t=aTracklet.t();

	  for (unsigned int j=0;j<D->stubs_[jSector].size();j++) {
	    double r=D->stubs_[jSector][j].r();
	    double z=D->stubs_[jSector][j].z();
	    double phi=D->stubs_[jSector][j].phi();
	    //cout << "r1 phi1 r2 phi2:"
	    //  <<stubs_[iSector][i].r()<<" "
	    //  <<stubs_[iSector][i].phi()<<" "
	    //  <<L->stubs_[jSector][j].r()<<" "
	    //  <<L->stubs_[jSector][j].phi()<<endl;

	    
	    double phiproj=phi0-0.5*(z-z0)*rinv/t;
	    double rproj=2.0*sin(0.5*(z-z0)*rinv/t)/rinv;

	    double deltaphi=phi-phiproj;
	    //cout << "deltaphi phi phiproj:"<<deltaphi<<" "<<phi<<" "<<phiproj<<" "<<phi0<<" "<<asin(0.5*r*rinv)<<endl;

	    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	    assert(fabs(deltaphi)<0.5*two_pi);

	    double rdeltaphi=r*deltaphi;
            double deltar=r-rproj;

	    if (fabs(rdeltaphi)>1.0) continue;
	    if (fabs(deltar)>2.0) continue;

	    //cout << "rdeltaphi deltar:"<<rdeltaphi<<" "<<deltar<<endl;

	    tracklets_[iSector][i].addStub(D->stubs_[jSector][j]);

	  }
	}
      }
    }
  }



private:

  double rmin_;
  double rmax_;
  double zmax_;

};



#endif



