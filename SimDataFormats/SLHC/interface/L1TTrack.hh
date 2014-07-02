#ifndef L1TTRACK_H
#define L1TTRACK_H

#include <iostream>
#include <assert.h>


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

    double largestresid;
    int ilargestresid;

    for (int i=0;i<1;i++){

      if (i>0) {
	rinv_=rinvfit4par_;
	phi0_=phi0fit4par_;
	z0_=z0fit4par_;
	t_=tfit4par_;
      }

      calculateDerivatives(false);
      linearTrackFit(false);
     

    }

    largestresid=-1.0;
    ilargestresid=-1;

    residuals(largestresid,ilargestresid);

    //cout << "Chisq largestresid: "<<chisq()<<" "<<largestresid<<endl;

    if (stubs_.size()>3&&chisq4par()>100.0&&largestresid>5.0) {
      //cout << "Refitting track"<<endl;
      stubs_.erase(stubs_.begin()+ilargestresid);
      rinv_=rinvfit4par_;
      phi0_=phi0fit4par_;
      z0_=z0fit4par_;
      t_=tfit4par_;
      calculateDerivatives(false);
      linearTrackFit(false);
      residuals(largestresid,ilargestresid);
    }

    calculateDerivatives(true);
    linearTrackFit(true);


  }


  void invert(double M[5][10],unsigned int n){

    assert(n<=5);

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



  void calculateDerivatives(bool withd0=true){

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
	D_[4][j]=1.0/sigmax;
	j++;
	//second the z position
	D_[0][j]=0.0;
	D_[1][j]=0.0;
	D_[2][j]=(2/rinv_)*asin(0.5*ri*rinv_)/sigmaz;
	D_[3][j]=1.0/sigmaz;
	D_[4][j]=0.0;
	j++;
      }
      else {
	//here we handle a disk hit
	//first we have the r position

	double r_track=2.0*sin(0.5*rinv_*(zi-z0_)/t_)/rinv_;
	double phi_track=phi0_-0.5*rinv_*(zi-z0_)/t_;

	int iphi=stubs_[i].iphi();
	double phii=stubs_[i].phi();

	double width=4.608;
	double nstrip=508.0;
	if (ri<60.0) {
	  width=4.8;
	  nstrip=480;
	}
	double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...

	if (stubs_[i].z()>0.0) Deltai=-Deltai;
	double theta0=asin(Deltai/ri);

	double rmultiplier=-sin(theta0-(phi_track-phii));
	double phimultiplier=r_track*cos(theta0-(phi_track-phii));


	double drdrinv=-2.0*sin(0.5*rinv_*(zi-z0_)/t_)/(rinv_*rinv_)
			+(zi-z0_)*cos(0.5*rinv_*(zi-z0_)/t_)/(rinv_*t_);
	double drdphi0=0;
	double drdt=-(zi-z0_)*cos(0.5*rinv_*(zi-z0_)/t_)/(t_*t_);
	double drdz0=-cos(0.5*rinv_*(zi-z0_)/t_)/t_;

	double dphidrinv=-0.5*(zi-z0_)/t_;
	double dphidphi0=1.0;
	double dphidt=0.5*rinv_*(zi-z0_)/(t_*t_);
	double dphidz0=0.5*rinv_/t_;
	
	D_[0][j]=drdrinv/sigmaz;
	D_[1][j]=drdphi0/sigmaz;
	D_[2][j]=drdt/sigmaz;
	D_[3][j]=drdz0/sigmaz;
	D_[4][j]=0;
	j++;
	//second the rphi position
	D_[0][j]=(phimultiplier*dphidrinv+rmultiplier*drdrinv)/sigmax;
	D_[1][j]=(phimultiplier*dphidphi0+rmultiplier*drdphi0)/sigmax;
	D_[2][j]=(phimultiplier*dphidt+rmultiplier*drdt)/sigmax;
	D_[3][j]=(phimultiplier*dphidz0+rmultiplier*drdz0)/sigmax;
	D_[4][j]=1.0/sigmax;
        //old calculation
	//D_[0][j]=-0.5*(zi-z0_)/(t_*(sigmax/ri));
	//D_[1][j]=1.0/(sigmax/ri);
	//D_[2][j]=-0.5*rinv_*(zi-z0_)/(t_*t_*(sigmax/ri));
	//D_[3][j]=0.5*rinv_/((sigmax/ri)*t_);
	j++;
      }

      //cout << "Exact rinv derivative: "<<i<<" "<<D_[0][j-2]<<" "<<D_[0][j-1]<<endl;
      //cout << "Exact phi0 derivative: "<<i<<" "<<D_[1][j-2]<<" "<<D_[1][j-1]<<endl;
      //cout << "Exact t derivative   : "<<i<<" "<<D_[2][j-2]<<" "<<D_[2][j-1]<<endl;
      //cout << "Exact z0 derivative  : "<<i<<" "<<D_[3][j-2]<<" "<<D_[3][j-1]<<endl;
	
	
    }
    
    //cout << "D:"<<endl;
    //for(unsigned int j=0;j<2*n;j++){
    //  cout <<D_[0][j]<<" "<<D_[1][j]<<" "<<D_[2][j]<<" "<<D_[3][j]<<endl;
    //}

     

    unsigned int npar=4;
    if (withd0) npar++;

    for(unsigned int i1=0;i1<npar;i1++){
      for(unsigned int i2=0;i2<npar;i2++){
	M_[i1][i2]=0.0;
	for(unsigned int j=0;j<2*n;j++){
	  M_[i1][i2]+=D_[i1][j]*D_[i2][j];	  
	}
      }
    }

    invert(M_,npar);

    for(unsigned int j=0;j<2*n;j++) {
      for(unsigned int i1=0;i1<npar;i1++) {
	MinvDt_[i1][j]=0.0;
	for(unsigned int i2=0;i2<npar;i2++) {
	  MinvDt_[i1][j]+=M_[i1][i2+npar]*D_[i2][j];
	}
      }
    }

  }

  void residuals(double& largestresid,int& ilargestresid) {

    unsigned int n=stubs_.size();

    //Next calculate the residuals

    double delta[40];

    double chisq=0.0;

    unsigned int j=0;

    bool print=false;

    if (print) cout << "Residuals ("<<chisq4par_<<") ["<<0.003*3.8/rinvfit4par_<<"]: ";

    largestresid=-1.0;
    ilargestresid=-1;

    for(unsigned int i=0;i<n;i++) {
      double ri=stubs_[i].r();
      double zi=stubs_[i].z();
      double phii=stubs_[i].phi();
      double sigmax=stubs_[i].sigmax();
      double sigmaz=stubs_[i].sigmaz();

      int layer=stubs_[i].layer();

      if (layer<1000) {
        //we are dealing with a barrel stub


	double deltaphi=phi0fit4par_-asin(0.5*ri*rinvfit4par_)-phii;
	if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	//cout << "4par : "<<phi0fit4par_<<" "<<rinvfit4par_<<" "<<deltaphi<<endl;
	assert(fabs(deltaphi)<0.1*two_pi);

	delta[j++]=ri*deltaphi/sigmax;
	delta[j++]=(z0fit4par_+(2.0/rinvfit4par_)*tfit4par_*asin(0.5*ri*rinvfit4par_)-zi)/sigmaz;
	
      }
      else {
	//we are dealing with a disk hit

	double r_track=2.0*sin(0.5*rinvfit4par_*(zi-z0fit4par_)/tfit4par_)/rinvfit4par_;
	double phi_track=phi0fit4par_-0.5*rinvfit4par_*(zi-z0fit4par_)/tfit4par_;

	int iphi=stubs_[i].iphi();

	double width=4.608;
	double nstrip=508.0;
	if (ri<60.0) {
	  width=4.8;
	  nstrip=480;
	}
	double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...

	if (stubs_[i].z()>0.0) Deltai=-Deltai;

	double theta0=asin(Deltai/ri);

	double Delta=Deltai-r_track*sin(theta0-(phi_track-phii));

	delta[j++]=(r_track-ri)/sigmaz;
	delta[j++]=Delta/sigmax;
      }

      if (fabs(delta[j-2])>largestresid) {
	largestresid=fabs(delta[j-2]);
	ilargestresid=i;
      }

      if (fabs(delta[j-1])>largestresid) {
	largestresid=fabs(delta[j-1]);
	ilargestresid=i;
      }
      
      if (print) cout << delta[j-2]<<" "<<delta[j-1]<<" ";

      chisq+=delta[j-2]*delta[j-2]+delta[j-1]*delta[j-1];

    }

    if (print) cout <<" ("<<chisq<<")"<<endl;

  }
  

  void linearTrackFit(bool withd0=true) {

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


	//numerical derivative check

	for (int iii=0;iii<0;iii++){

	  double drinv=0.0;
	  double dphi0=0.0;
	  double dt=0.0;
	  double dz0=0.0;

	  if (iii==0) drinv=0.001*fabs(rinv_);
	  if (iii==1) dphi0=0.001;
	  if (iii==2) dt=0.001;
	  if (iii==3) dz0=0.01;

	  double deltaphi=phi0_+dphi0-asin(0.5*ri*(rinv_+drinv))-phii;
	  if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	  if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	  assert(fabs(deltaphi)<0.1*two_pi);

	  double delphi=ri*deltaphi/sigmax;
	  double deltaz=(z0_+dz0+(2.0/(rinv_+drinv))*(t_+dt)*asin(0.5*ri*(rinv_+drinv))-zi)/sigmaz;


	  if (iii==0) cout << "Numerical rinv derivative: "<<i<<" "
			   <<(delphi-delta[j-2])/drinv<<" "
			   <<(deltaz-delta[j-1])/drinv<<endl;

	  if (iii==1) cout << "Numerical phi0 derivative: "<<i<<" "
			   <<(delphi-delta[j-2])/dphi0<<" "
			   <<(deltaz-delta[j-1])/dphi0<<endl;

	  if (iii==2) cout << "Numerical t derivative: "<<i<<" "
			   <<(delphi-delta[j-2])/dt<<" "
			   <<(deltaz-delta[j-1])/dt<<endl;

	  if (iii==3) cout << "Numerical z0 derivative: "<<i<<" "
			   <<(delphi-delta[j-2])/dz0<<" "
			   <<(deltaz-delta[j-1])/dz0<<endl;

	}



      }
      else {
	//we are dealing with a disk hit

	double r_track=2.0*sin(0.5*rinv_*(zi-z0_)/t_)/rinv_;
	//cout <<"t_track 1: "<<r_track<<endl;
	double phi_track=phi0_-0.5*rinv_*(zi-z0_)/t_;

	int iphi=stubs_[i].iphi();

	double width=4.608;
	double nstrip=508.0;
	if (ri<60.0) {
	  width=4.8;
	  nstrip=480;
	}
	double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...

	if (stubs_[i].z()>0.0) Deltai=-Deltai;

	double theta0=asin(Deltai/ri);

	double Delta=Deltai-r_track*sin(theta0-(phi_track-phii));

	delta[j++]=(r_track-ri)/sigmaz;
	//double deltaphi=phi_track-phii;
	//if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	//if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	//assert(fabs(deltaphi)<0.1*two_pi);
	//delta[j++]=deltaphi/(sigmax/ri);
	delta[j++]=Delta/sigmax;

	//numerical derivative check

	for (int iii=0;iii<0;iii++){

	  double drinv=0.0;
	  double dphi0=0.0;
	  double dt=0.0;
	  double dz0=0.0;

	  if (iii==0) drinv=0.001*fabs(rinv_);
	  if (iii==1) dphi0=0.001;
	  if (iii==2) dt=0.001;
	  if (iii==3) dz0=0.01;

	  r_track=2.0*sin(0.5*(rinv_+drinv)*(zi-(z0_+dz0))/(t_+dt))/(rinv_+drinv);
	  //cout <<"t_track 2: "<<r_track<<endl;
	  phi_track=phi0_+dphi0-0.5*(rinv_+drinv)*(zi-(z0_+dz0))/(t_+dt);
	  
	  iphi=stubs_[i].iphi();

	  double width=4.608;
	  double nstrip=508.0;
	  if (ri<60.0) {
	    width=4.8;
	    nstrip=480;
	  }
	  Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...

	  if (stubs_[i].z()>0.0) Deltai=-Deltai;
	  theta0=asin(Deltai/ri);
	  
	  Delta=Deltai-r_track*sin(theta0-(phi_track-phii));

	  if (iii==0) cout << "Numerical rinv derivative: "<<i<<" "
			   <<((r_track-ri)/sigmaz-delta[j-2])/drinv<<" "
			   <<(Delta/sigmax-delta[j-1])/drinv<<endl;

	  if (iii==1) cout << "Numerical phi0 derivative: "<<i<<" "
			   <<((r_track-ri)/sigmaz-delta[j-2])/dphi0<<" "
			   <<(Delta/sigmax-delta[j-1])/dphi0<<endl;

	  if (iii==2) cout << "Numerical t derivative: "<<i<<" "
			   <<((r_track-ri)/sigmaz-delta[j-2])/dt<<" "
			   <<(Delta/sigmax-delta[j-1])/dt<<endl;

	  if (iii==3) cout << "Numerical z0 derivative: "<<i<<" "
			   <<((r_track-ri)/sigmaz-delta[j-2])/dz0<<" "
			   <<(Delta/sigmax-delta[j-1])/dz0<<endl;

	}

      }

      chisq+=(delta[j-2]*delta[j-2]+delta[j-1]*delta[j-1]);

    }

    double drinv=0.0;
    double dphi0=0.0;
    double dd0=0.0;
    double dt=0.0;
    double dz0=0.0;

    double drinv_cov=0.0;
    double dphi0_cov=0.0;
    double dd0_cov=0.0;
    double dt_cov=0.0;
    double dz0_cov=0.0;



    for(unsigned int j=0;j<2*n;j++) {
      drinv-=MinvDt_[0][j]*delta[j];
      //cout << "MinvDt_[0][j] delta[j]:"<<MinvDt_[0][j]<<" "<<delta[j]<<endl;
      dphi0-=MinvDt_[1][j]*delta[j];
      dt-=MinvDt_[2][j]*delta[j];
      dz0-=MinvDt_[3][j]*delta[j];
      if (withd0) dd0-=MinvDt_[4][j]*delta[j];

      drinv_cov+=D_[0][j]*delta[j];
      dphi0_cov+=D_[1][j]*delta[j];
      dt_cov+=D_[2][j]*delta[j];
      dz0_cov+=D_[3][j]*delta[j];
      if (withd0) dd0_cov+=D_[4][j]*delta[j];
    }
    

    double deltaChisq=drinv*drinv_cov+dphi0*dphi0_cov+dt*dt_cov+dz0*dz0_cov;
    if (withd0) deltaChisq+=dd0*dd0_cov;

    //drinv=0.0; dphi0=0.0; dt=0.0; dz0=0.0;

    if (withd0) {
      rinvfit_=rinv_+drinv;
      phi0fit_=phi0_+dphi0;
      
      tfit_=t_+dt;
      z0fit_=z0_+dz0;
      
      d0fit_=dd0;

      chisq_=(chisq+deltaChisq);

    } else {
      rinvfit4par_=rinv_+drinv;
      phi0fit4par_=phi0_+dphi0;
      
      tfit4par_=t_+dt;
      z0fit4par_=z0_+dz0;
      
      chisq4par_=(chisq+deltaChisq);
    }

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

  int npixelstrip() const {
    int count=0;
    for (unsigned int i=0;i<stubs_.size();i++){
      if (stubs_[i].sigmaz()<0.5) count++; 
    }
    return count;
  }

  L1TTracklet getSeed() const { return seed_; }
  vector<L1TStub> getStubs() const { return stubs_; }
  unsigned int nstub() const { return stubs_.size(); }
  double getRinv() const { return rinv_; }
  double getPhi0() const { return phi0_; }
  double getZ0() const { return z0_; }
  double getT() const { return t_; }
  bool isCombinatorics() const { return isCombinatorics_; }
  double getSimTrackID() const { return SimTrackID_; }

  double pt(double bfield) const { return 0.00299792*bfield/rinvfit_; }
  double pt4par(double bfield) const { return 0.00299792*bfield/rinvfit4par_; }
  //double ipt(double bfield) const { return 0.00299792*bfield/irinvfit(); }
  double ptseed(double bfield) const { return 0.00299792*bfield/rinv_; }

  double rinv() const { return rinvfit_; }
  double rinv4par() const { return rinvfit4par_; }
  double phi04par() const { return phi0fit4par_;}
  double phi0() const { return phi0fit_;}
  double d0() const { return d0fit_;}
  //double iphi0() const { return iphi0fit();}
  double phi0seed() const { return phi0_;}

  double eta() const { static double two_pi=8*atan(1.0);
    return -log(tan(0.5*(0.25*two_pi-atan(tfit_)))); }
  double eta4par() const { static double two_pi=8*atan(1.0);
    return -log(tan(0.5*(0.25*two_pi-atan(tfit4par_)))); }
  //double ieta() const { static double two_pi=8*atan(1.0);
  //  return -log(tan(0.5*(0.25*two_pi-atan(itfit())))); }
  double etaseed() const { static double two_pi=8*atan(1.0);
    return -log(tan(0.5*(0.25*two_pi-atan(t_)))); }

  double z0() const { return z0fit_; }
  double z04par() const { return z0fit4par_; }
  //double iz0() const { return iz0fit(); }
  double z0seed() const { return z0_; }

  double chisq() const {return chisq_;}
  double chisq4par() const {return chisq4par_;}

  double chisqdof() const {return chisq_/(2*stubs_.size()-5);}
  double chisqdof4par() const {return chisq4par_/(2*stubs_.size()-4);}
  


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
  double d0fit_;
  double z0fit_;
  double tfit_;

  double rinvfit4par_;
  double phi0fit4par_;
  double z0fit4par_;
  double tfit4par_;

  double chisq_;
  double chisq4par_;

  int ichisq1_;
  int ichisq2_;

  double D_[5][40];
  
  double M_[5][10];
  
  double MinvDt_[5][40];


};


#endif
