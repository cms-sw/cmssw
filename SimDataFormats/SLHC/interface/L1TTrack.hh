#ifndef L1TTRACK_H
#define L1TTRACK_H

#include <iostream>
#include <assert.h>

#include "L1TConstants.hh"

using namespace std;

class L1TTrack{

public:

  L1TTrack() { };

  L1TTrack(const L1TTracklet& seed) {

    seed_=seed;
    rinv_=seed.rinv();
    phi0_=seed.phi0();
    z0_=seed.z0();
    t_=seed.t();
    stubs_=seed.getStubs();

    //double frac;

    //cout << "Constructed tracks with "<<stubs_.size()<<" stubs and simtrackid="
    //	 << simtrackid(frac)<<endl;

    calculateDerivatives();

    linearTrackFit();

  }


  void invert(double M[4][8],unsigned int n){

    assert(n<=4);

    unsigned int i,j,k;
    double ratio,a;

    for(i = 0; i < n; i++){
      for(j = n; j < 2*n; j++){
	if(i==(j-n))
	  M[i][j] = 1.0;
	else
	  M[i][j] = 0.0;
      }
    }

    for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
	if(i!=j){
	  ratio = M[j][i]/M[i][i];
	  for(k = 0; k < 2*n; k++){
	    M[j][k] -= ratio * M[i][k];
	  }
	}
      }
    }

    for(i = 0; i < n; i++){
      a = M[i][i];
      for(j = 0; j < 2*n; j++){
	M[i][j] /= a;
      }
    }
  }



  void calculateDerivatives(){

    unsigned int n=stubs_.size();

    assert(n<=20);


    int j=0;

    for(unsigned int i=0;i<n;i++) {

      double ri=stubs_[i].r();
      double zi=stubs_[i].z();
      double sigmax=stubs_[i].sigmax();
      double sigmaz=stubs_[i].sigmaz();

      //cout << "i layer "<<i<<" "<<stubs_[i].layer()<<endl;

      if (stubs_[i].layer()<1000){
	//here we handle a barrel hit

	//first we have the phi position
	D_[0][j]=-0.5*ri*ri/sqrt(1-0.25*ri*ri*rinv_*rinv_)/sigmax;
	D_[1][j]=ri/sigmax;
	D_[2][j]=0.0;
	D_[3][j]=0.0;
	j++;
	//second the z position
	D_[0][j]=0.0;
	D_[1][j]=0.0;
	D_[2][j]=(2/rinv_)*asin(0.5*ri*rinv_)/sigmaz;
	D_[3][j]=1.0/sigmaz;
	j++;
      }
      else {
	//here we handle a disk hit
	//first we have the r position
	D_[0][j]=(-2.0*sin(0.5*rinv_*(zi-z0_)/t_)
		  +(zi-z0_)*cos(0.5*rinv_*(zi-z0_)/t_)/(rinv_*t_))/sigmaz;
	D_[1][j]=0.0;
	D_[2][j]=-(zi-z0_)*cos(0.5*rinv_*(zi-z0_)/t_)/(sigmaz*t_*t_);
	D_[3][j]=-cos(0.5*rinv_*(zi-z0_)/t_)/(sigmaz*t_);
	j++;
	//second the phi position
	D_[0][j]=-0.5*(zi-z0_)/(t_*(sigmax/ri));
	D_[1][j]=1.0/(sigmax/ri);
	D_[2][j]=-0.5*rinv_*(zi-z0_)/(t_*t_*(sigmax/ri));
	D_[3][j]=0.5*rinv_/((sigmax/ri)*t_);
	j++;
      }
	
	
    }
    
    //cout << "D:"<<endl;
    //for(unsigned int j=0;j<2*n;j++){
    //  cout <<D_[0][j]<<" "<<D_[1][j]<<" "<<D_[2][j]<<" "<<D_[3][j]<<endl;
    //}

     



    for(unsigned int i1=0;i1<4;i1++){
      for(unsigned int i2=0;i2<4;i2++){
	M_[i1][i2]=0.0;
	for(unsigned int j=0;j<2*n;j++){
	  M_[i1][i2]+=D_[i1][j]*D_[i2][j];	  
	}
      }
    }

    invert(M_,4);

    for(unsigned int j=0;j<2*n;j++) {
      for(unsigned int i1=0;i1<4;i1++) {
	MinvDt_[i1][j]=0.0;
	for(unsigned int i2=0;i2<4;i2++) {
	  MinvDt_[i1][j]+=M_[i1][i2+4]*D_[i2][j];
	}
      }
    }

  }

  void linearTrackFit() {

    unsigned int n=stubs_.size();

    //Next calculate the residuals

    double delta[40];

    double chisq=0;

    unsigned int j=0;

    for(unsigned int i=0;i<n;i++) {
      double ri=stubs_[i].r();
      double zi=stubs_[i].z();
      double phii=stubs_[i].phi();
      double sigmax=stubs_[i].sigmax();
      double sigmaz=stubs_[i].sigmaz();

      int layer=stubs_[i].layer();

      if (layer<1000) {
        //we are dealing with a barrel stub

	double deltaphi=phi0_-asin(0.5*ri*rinv_)-phii;
	if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	assert(fabs(deltaphi)<0.1*two_pi);

	delta[j++]=ri*deltaphi/sigmax;
	delta[j++]=(z0_+(2.0/rinv_)*t_*asin(0.5*ri*rinv_)-zi)/sigmaz;
	//cout << "delta[j-2] delta[j-1]:"<<delta[j-2]<<" "<<delta[j-1]<<endl;

      }
      else {
	//we are dealing with a disk hit

	delta[j++]=(2.0*sin(0.5*rinv_*(zi-z0_)/t_)/rinv_-ri)/sigmaz;
	double deltaphi=phi0_-0.5*rinv_*(zi-z0_)/t_-phii;
	if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	assert(fabs(deltaphi)<0.1*two_pi);
	delta[j++]=deltaphi/(sigmax/ri);

      }

      chisq+=(delta[j-2]*delta[j-2]+delta[j-1]*delta[j-1]);

    }

    double drinv=0.0;
    double dphi0=0.0;
    double dt=0.0;
    double dz0=0.0;

    double drinv_cov=0.0;
    double dphi0_cov=0.0;
    double dt_cov=0.0;
    double dz0_cov=0.0;



    for(unsigned int j=0;j<2*n;j++) {
      drinv-=MinvDt_[0][j]*delta[j];
      //cout << "MinvDt_[0][j] delta[j]:"<<MinvDt_[0][j]<<" "<<delta[j]<<endl;
      dphi0-=MinvDt_[1][j]*delta[j];
      dt-=MinvDt_[2][j]*delta[j];
      dz0-=MinvDt_[3][j]*delta[j];

      drinv_cov+=D_[0][j]*delta[j];
      dphi0_cov+=D_[1][j]*delta[j];
      dt_cov+=D_[2][j]*delta[j];
      dz0_cov+=D_[3][j]*delta[j];
    }
    

    double deltaChisq=drinv*drinv_cov+dphi0*dphi0_cov+dt*dt_cov+dz0*dz0_cov;

    //drinv=0.0; dphi0=0.0; dt=0.0; dz0=0.0;

    rinvfit_=rinv_+drinv;
    phi0fit_=phi0_+dphi0;

    tfit_=t_+dt;
    z0fit_=z0_+dz0;

    chisq1_=(chisq+deltaChisq);
    chisq2_=0.0;

    //cout << "Trackfit:"<<endl;
    //cout << "rinv_ drinv: "<<rinv_<<" "<<drinv<<endl;
    //cout << "phi0_ dphi0: "<<phi0_<<" "<<dphi0<<endl;
    //cout << "t_ dt      : "<<t_<<" "<<dt<<endl;
    //cout << "z0_ dz0    : "<<z0_<<" "<<dz0<<endl;

  }



  bool overlap(const L1TTrack& aTrack) const {
    
    int nSame=0;
    for(unsigned int i=0;i<stubs_.size();i++) {
      for(unsigned int j=0;j<aTrack.stubs_.size();j++) {
	if (stubs_[i]==aTrack.stubs_[j]) nSame++;
      }
    }

    return (nSame>=2);

  }


  int simtrackid(double& fraction) const {

    //cout << "In L1TTrack::simtrackid"<<endl;

    map<int, int> simtrackids;

    for(unsigned int i=0;i<stubs_.size();i++){
      //cout << "Stub simtrackid="<<stubs_[i].simtrackid()<<endl;
      simtrackids[stubs_[i].simtrackid()]++;
    }

    int simtrackid=0;
    int nsimtrack=0;

    map<int, int>::const_iterator it=simtrackids.begin();

    while(it!=simtrackids.end()) {
      //cout << it->first<<" "<<it->second<<endl;
      if (it->second>nsimtrack) {
	nsimtrack=it->second;
	simtrackid=it->first;
      }
      it++;
    }

    //cout << "L1TTrack::simtrackid done"<<endl;

    fraction=(1.0*nsimtrack)/stubs_.size();

    return simtrackid;

  }


  L1TTracklet getSeed() const { return seed_; }
  vector<L1TStub> getStubs() const { return stubs_; }
  unsigned int nstub() const { return stubs_.size(); }
  double rinv() const { return rinv_; }
  double getPhi0() const { return phi0_; }
  double getZ0() const { return z0_; }
  double getT() const { return t_; }
  bool isCombinatorics() const { return isCombinatorics_; }
  double getSimTrackID() const { return SimTrackID_; }

  double pt(double bfield) const { return 0.00299792*bfield/rinvfit_; }
  //double ipt(double bfield) const { return 0.00299792*bfield/irinvfit(); }
  double ptseed(double bfield) const { return 0.00299792*bfield/rinv_; }

  double phi0() const { return phi0fit_;}
  //double iphi0() const { return iphi0fit();}
  double phi0seed() const { return phi0_;}

  double eta() const { static double two_pi=8*atan(1.0);
    return -log(tan(0.5*(0.25*two_pi-atan(tfit_)))); }
  //double ieta() const { static double two_pi=8*atan(1.0);
  //  return -log(tan(0.5*(0.25*two_pi-atan(itfit())))); }
  double etaseed() const { static double two_pi=8*atan(1.0);
    return -log(tan(0.5*(0.25*two_pi-atan(t_)))); }

  double z0() const { return z0fit_; }
  //double iz0() const { return iz0fit(); }
  double z0seed() const { return z0_; }

  double chisq1() const {return chisq1_;}
  double chisq2() const {return chisq2_;}

  double chisq1dof() const {return chisq1_/(stubs_.size()-2);}
  double chisq2dof() const {return chisq2_/(stubs_.size()-2);}
  
  double chisq() const {return chisq1_+chisq2_; }
  double chisqdof() const {return (chisq1_+chisq2_)/(2*stubs_.size()-4); }


private:

  L1TTracklet seed_;
  vector<L1TStub> stubs_;
  double rinv_;
  double phi0_;
  double z0_;
  double t_;
  bool isCombinatorics_;
  int SimTrackID_;
  double rinvfit_;
  double phi0fit_;
  double z0fit_;
  double tfit_;

  int irinvfit_;
  int iphi0fit_;
  int iz0fit_;
  int itfit_;

  double chisq1_;
  double chisq2_;

  int ichisq1_;
  int ichisq2_;

  double D_[4][40];
  
  double M_[4][8];
  
  double MinvDt_[4][40];


};


#endif
