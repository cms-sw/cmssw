#ifndef L1TTRACK_H
#define L1TTRACK_H

#include <iostream>
#include <assert.h>

#include "L1TConstants.hh"
#include "L1TWord.hh"

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
    stubs_=seed.getAllStubs();

    //fitTrack();

    //cout << "Starting track fit ---------------"<<endl;

    calculateDerivatives();

    linearTrackFit();

    linearTrackFitBinary();

  }


  void invert(double M[3][6],unsigned int n){

    assert(n<=3);

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


    for(unsigned int i=0;i<n;i++) {
      double ri=stubs_[i].r();
      
      Drphi_[0][i]=-0.5*ri*ri/sqrt(1-0.25*ri*ri*rinv_*rinv_);
      Drphi_[1][i]=ri;

      Drz_[0][i]=(2/rinv_)*asin(0.5*ri*rinv_);
      Drz_[1][i]=1.0;
    }



    for(unsigned int i1=0;i1<2;i1++){
      for(unsigned int i2=0;i2<2;i2++){
	Mrphi_[i1][i2]=0.0;
	Mrz_[i1][i2]=0.0;
	for(unsigned int j=0;j<n;j++){
	  Mrphi_[i1][i2]+=Drphi_[i1][j]*Drphi_[i2][j];	  
	  Mrz_[i1][i2]+=Drz_[i1][j]*Drz_[i2][j];	  
	}
      }
    }

    invert(Mrphi_,2);
    invert(Mrz_,2);




    for(unsigned int j=0;j<n;j++) {
      for(unsigned int i1=0;i1<2;i1++) {
	MinvDtrphi_[i1][j]=0.0;
	MinvDtrz_[i1][j]=0.0;
	for(unsigned int i2=0;i2<2;i2++) {
	  MinvDtrphi_[i1][j]+=Mrphi_[i1][i2+2]*Drphi_[i2][j];
	  MinvDtrz_[i1][j]+=Mrz_[i1][i2+2]*Drz_[i2][j];
	}
      }
    }

    for(unsigned int j=0;j<n;j++) {
      iMinvDtrphi_[0][j]=MinvDtrphi_[0][j]*BASE10;
      iMinvDtrphi_[1][j]=MinvDtrphi_[1][j]*BASE11;
      iMinvDtrz_[0][j]=MinvDtrz_[0][j]*BASE12;
      iMinvDtrz_[1][j]=MinvDtrz_[1][j]*BASE13;

      //cout << "iMinvDtrphi_[0][j]:"<<iMinvDtrphi_[0][j].value()<<endl;
      //cout << "iMinvDtrphi_[1][j]:"<<iMinvDtrphi_[1][j].value()<<endl;
      //cout << "iMinvDtrz_[0][j]:"<<iMinvDtrz_[0][j].value()<<endl;
      //cout << "iMinvDtrz_[1][j]:"<<iMinvDtrz_[1][j].value()<<endl;


      iDrphi_[0][j]=Drphi_[0][j]/BASE14;
      iDrphi_[1][j]=Drphi_[1][j]*BASE15;
      iDrz_[0][j]=Drz_[0][j]*BASE16;
      iDrz_[1][j]=Drz_[1][j]*BASE17;

      //cout << "iDrphi_[0][j]:"<<iDrphi_[0][j].value()<<endl;
      //cout << "iDrphi_[1][j]:"<<iDrphi_[1][j].value()<<endl;
      //cout << "iDrz_[0][j]:"<<iDrz_[0][j].value()<<endl;
      //cout << "iDrz_[1][j]:"<<iDrz_[1][j].value()<<endl;

      
    }

  }


  void linearTrackFitBinary(){


    unsigned int n=stubs_.size();

    //Next calculate the residuals

    static vector<L1TWord> ideltarphim(20);
    static vector<L1TWord> ideltazm(20);

    static bool first=true;
    
    if (first) {
      first=false;
      ideltarphim[0].setName("ideltarphim[0]",true);
      ideltarphim[1].setName("ideltarphim[1]",true);
      ideltarphim[2].setName("ideltarphim[2]",true);
      ideltarphim[3].setName("ideltarphim[3]",true);
      ideltarphim[4].setName("ideltarphim[4]",true);
      ideltarphim[5].setName("ideltarphim[5]",true);
      ideltarphim[6].setName("ideltarphim[6]",true);
      ideltarphim[7].setName("ideltarphim[7]",true);
      ideltarphim[8].setName("ideltarphim[8]",true);
      ideltarphim[9].setName("ideltarphim[9]",true);
      ideltarphim[10].setName("ideltarphim[10]",true);
      ideltarphim[11].setName("ideltarphim[11]",true);
      ideltarphim[12].setName("ideltarphim[12]",true);

      ideltazm[0].setName("ideltazm[0]",true);
      ideltazm[1].setName("ideltazm[1]",true);
      ideltazm[2].setName("ideltazm[2]",true);
      ideltazm[3].setName("ideltazm[3]",true);
      ideltazm[4].setName("ideltazm[4]",true);
      ideltazm[5].setName("ideltazm[5]",true);
      ideltazm[6].setName("ideltazm[6]",true);
      ideltazm[7].setName("ideltazm[7]",true);
      ideltazm[8].setName("ideltazm[8]",true);
      ideltazm[9].setName("ideltazm[9]",true);
      ideltazm[10].setName("ideltazm[10]",true);
      ideltazm[11].setName("ideltazm[11]",true);
      ideltazm[12].setName("ideltazm[12]",true);

    }
    
    static L1TWord ichisq_rphi(MAXINT,"ichisq_rphi",true);
    static L1TWord ichisq_rz(MAXINT,"ichisq_rz",true);

    ichisq_rphi=0;
    ichisq_rz=0;

    for(unsigned int i=0;i<n;i++) {

      ideltarphim[i]=stubs_[i].ideltarphi();
      ideltazm[i]=stubs_[i].ideltaz();

      //cout << "i ideltarphim ideltaz : "<<i<<" "<<ideltarphim[i].value()<<" "<<ideltazm[i].value()<<" "<<ideltarphim[i].value()*DX*BASE18/BASE<<" "<<ideltazm[i].value()*DZ<<endl;

      ichisq_rphi+=ideltarphim[i]*ideltarphim[i];
      ichisq_rz+=ideltazm[i]*ideltazm[i];

    }


    static L1TWord idrinv(MAXINT,"idrinv",true);
    static L1TWord idphi0(MAXINT,"idphi0",true);
    static L1TWord idt(MAXINT,"idt",true);
    static L1TWord idz0(MAXINT,"idz0",true);

    static L1TWord idrinv_cov(MAXINT,"idrinv_cov",true);
    static L1TWord idphi0_cov(MAXINT,"idphi0_cov",true);
    static L1TWord idt_cov(MAXINT,"idt_cov",true);
    static L1TWord idz0_cov(MAXINT,"idz0_cov",true);

    idrinv=0;
    idphi0=0;
    idt=0;
    idz0=0;

    idrinv_cov=0;
    idphi0_cov=0;
    idt_cov=0;
    idz0_cov=0;


    for(unsigned int j=0;j<n;j++) {
      idrinv-=iMinvDtrphi_[0][j]*ideltarphim[j];
      idphi0-=iMinvDtrphi_[1][j]*ideltarphim[j];
      idt-=iMinvDtrz_[0][j]*ideltazm[j];
      idz0-=iMinvDtrz_[1][j]*ideltazm[j];

      idrinv_cov+=iDrphi_[0][j]*ideltarphim[j];
      idphi0_cov+=iDrphi_[1][j]*ideltarphim[j];
      idt_cov+=iDrz_[0][j]*ideltazm[j];
      idz0_cov+=iDrz_[1][j]*ideltazm[j];
    }


    //cout << "Linear ichisq_rhpi_ ichisq_rz :"<<ichisq_rphi.value()<<" "<<ichisq_rz.value()
    // << " "<<ichisq_rphi.value()*DX*DX*BASE18*BASE18/(sigmax*sigmax*BASE*BASE)
    // << " "<<ichisq_rz.value()*DZ*DZ/(sigmaz*sigmaz)<<endl;


    //qwerty

    //cout << "idrinv idphi0 idt idz0: "<<idrinv.value()<<" "<<idphi0.value()<<" "<<idt.value()<<" "<<idz0.value()<<endl;
    //cout << "idrinv_cov idphi0_cov idt_cov idz0_cov: "<<idrinv_cov.value()<<" "<<idphi0_cov.value()<<" "<<idt_cov.value()<<" "<<idz0_cov.value()<<endl;

    L1TWord ideltaChisqrphi=(idrinv/BASE19)*(idrinv_cov/(BASE10/BASE19))*BASE14+
      (idphi0/BASE19)*(idphi0_cov/((BASE11/BASE19)*BASE15));
    L1TWord ideltaChisqrz=(idt/BASE20)*(idt_cov/BASE16)/(BASE12/BASE20)+
      (idz0/BASE13)*(idz0_cov/BASE17);

    ichisq1_=(ichisq_rphi+ideltaChisqrphi).value();
    ichisq2_=(ichisq_rz+ideltaChisqrz).value();

    //cout << "Linear updates ichisq_rhpi_ ichisq_rz :"<<(ichisq_rphi+ideltaChisqrphi).value()<<" "<<(ichisq_rz+ideltaChisqrz).value()
    //	 << " "<<(ichisq_rphi+ideltaChisqrphi).value()*DX*DX*BASE18*BASE18/(sigmax*sigmax*BASE*BASE)
    //	 << " "<<(ichisq_rz+ideltaChisqrz).value()*DZ*DZ/(sigmaz*sigmaz)<<endl;

  
    irinvfit_=seed_.irinv()+(idrinv/((BASE10/BASE3)*(BASE/BASE18))).value();
    iphi0fit_=seed_.iphi0()+(idphi0.value()*DX*BASE18)/BASE11;
    itfit_=seed_.it()+idt.value()*DX*BASE3/BASE12;
    iz0fit_=seed_.iz0()+idz0.value()/BASE13;


    //cout << "idrinv="<<idrinv.value()<<" "<<(idrinv.value()*DX*BASE18)/BASE10/BASE<<endl;
    //cout << "idphi0="<<idphi0.value()<<" "<<(idphi0.value()*DX*BASE18)/BASE11/BASE<<endl;
    //cout << "idt="<<idt.value()<<" "<<idt.value()*DZ/BASE12<<endl;
    //cout << "idz0="<<idz0.value()<<" "<<idz0.value()*DZ/BASE13<<endl;

    //cout << "idrinv rescaled:"<<(idrinv/((BASE10/BASE3)*(BASE/BASE18))).value()
    //	 << " " << (idrinv/((BASE10/BASE3)*(BASE/BASE18))).value()*DX/BASE3<<endl;
    //cout << "idphi0 rescaled:"<<(idphi0.value()*DX*BASE18)/BASE11
    //	 << " " << (idphi0.value()*DX*BASE18)/BASE11/BASE<<endl;
    //cout << "idt rescaled:"<<idt.value()*DX*BASE3/BASE12
    //	 << " " << idt.value()*DX*BASE3/BASE12*DZ/DX/BASE3<<endl;
    //cout << "idz0 rescaled:"<<idz0.value()/BASE13
    //	 << " " << idz0.value()/BASE13*DZ<<endl;

  }



  void linearTrackFit() {

    unsigned int n=stubs_.size();

    //Next calculate the residuals

    double deltarphim[20];
    double deltazm[20];

    double chisq_rphi=0;
    double chisq_rz=0;

    for(unsigned int i=0;i<n;i++) {
      double ri=stubs_[i].r();
      double zi=stubs_[i].z();
      double phii=stubs_[i].phi();

      double deltaphi=phi0_-asin(0.5*ri*rinv_)-phii;
      if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
      if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
      assert(fabs(deltaphi)<0.1*two_pi);

      deltarphim[i]=ri*deltaphi;
      deltazm[i]=z0_+(2.0/rinv_)*t_*asin(0.5*ri*rinv_)-zi;

      chisq_rphi+=deltarphim[i]*deltarphim[i];
      chisq_rz+=deltazm[i]*deltazm[i];

      //cout << "i deltarphim deltaz : "<<i<<" "<<deltarphim[i]<<" "<<deltarphim[i]/DX*BASE/BASE18<<" "<<deltazm[i]<<endl;
    }

    double drinv=0.0;
    double dphi0=0.0;
    double dt=0.0;
    double dz0=0.0;

    double drinv_cov=0.0;
    double dphi0_cov=0.0;
    double dt_cov=0.0;
    double dz0_cov=0.0;



    for(unsigned int j=0;j<n;j++) {
      drinv-=MinvDtrphi_[0][j]*deltarphim[j];
      dphi0-=MinvDtrphi_[1][j]*deltarphim[j];
      dt-=MinvDtrz_[0][j]*deltazm[j];
      dz0-=MinvDtrz_[1][j]*deltazm[j];

      drinv_cov+=Drphi_[0][j]*deltarphim[j];
      dphi0_cov+=Drphi_[1][j]*deltarphim[j];
      dt_cov+=Drz_[0][j]*deltazm[j];
      dz0_cov+=Drz_[1][j]*deltazm[j];
    }


    //cout << "drinv="<<drinv<<endl;
    //cout << "dphi0="<<dphi0<<endl;
    //cout << "dt="<<dt<<endl;
    //cout << "dz0="<<dz0<<endl;

    //cout << "Linear chisq_rhpi_ chisq_rz :"<<chisq_rphi/(sigmax*sigmax)<<" "<<chisq_rz/(sigmaz*sigmaz)<<endl;

    double deltaChisqrphi=(drinv*drinv_cov+dphi0*dphi0_cov);
    double deltaChisqrz=(dt*dt_cov+dz0*dz0_cov);

    rinvfit_=rinv_+drinv;
    phi0fit_=phi0_+dphi0;

    tfit_=t_+dt;
    z0fit_=z0_+dz0;

    chisq1_=(chisq_rphi+deltaChisqrphi)/(sigmax*sigmax);
    chisq2_=(chisq_rz+deltaChisqrz)/(sigmaz*sigmaz);

    //cout << "Linear chisq1_ chisq2_ :"<<chisq1_<<" "<<chisq2_<<endl;

      //cout << "Linear drinv dphi0 dt dz0 :"<<drinv<<" "<<dphi0<<" "<<dt<<" "<<dz0<<endl;
      //cout << "Linear chisqrphi chisqrphifit chisqrz chisqrzfit :"
      //	 << chisq_rphi <<" "<<chisq_rphi+deltaChisqrphi<<" "
      // << chisq_rz <<" "<<chisq_rz+deltaChisqrz<<endl;
      

  }



  void fitTrack() {

    double sx=0.0;
    double sxx=0.0;
    double sxy=0.0;
    double sy=0.0;

    double zsx=0.0;
    double zsxx=0.0;
    double zsxy=0.0;
    double zsy=0.0;

    for(unsigned int i=0;i<stubs_.size();i++) {
      double ri=stubs_[i].r();
      double zi=stubs_[i].z();
      double phii=stubs_[i].phi();
      double xi=-0.5*ri/sqrt(1.0-0.25*ri*ri*rinv_*rinv_);
      double yi=-phi0_+phii+asin(0.5*ri*rinv_);
      //cout << "xi yi zi:"<<xi<<" "<<yi<<" "<<zi<<" "<<phii<<" "<<ri<<endl;
      sx+=xi;
      sxx+=xi*xi;
      sxy+=xi*yi;
      sy+=yi;
      double zxi=(2.0/rinv_)*asin(0.5*ri*rinv_);
      double zyi=zi-z0_-(2.0/rinv_)*t_*asin(0.5*ri*rinv_);
      //cout << "zxi zyi zi:"<<zxi<<" "<<zyi<<" "<<zi<<endl;
      zsx+=zxi;
      zsxx+=zxi*zxi;
      zsxy+=zxi*zyi;
      zsy+=zyi;
    }

    double s=stubs_.size();

    double drinv=(sx*sy-s*sxy)/(sx*sx-s*sxx);
    //cout << drinv<<" "<<sx<<" "<<sy<<" "<<s<<" "<<sxy<<" "<<sxx<<endl;
    double dphi0=(sx*sxy-sxx*sy)/(sx*sx-s*sxx);

    rinvfit_=rinv_+drinv;
    phi0fit_=phi0_+dphi0;

    double dt=(zsx*zsy-s*zsxy)/(zsx*zsx-s*zsxx);
    double dz0=(zsx*zsxy-zsxx*zsy)/(zsx*zsx-s*zsxx);

    tfit_=t_+dt;
    z0fit_=z0_+dz0;

    //cout << "Old drinv dphi0 dt dz0 :"<<drinv<<" "<<dphi0<<" "<<dt<<" "<<dz0<<endl;

    chisq1_=0.0;
    chisq2_=0.0;
    for(unsigned int i=0;i<stubs_.size();i++) {
      double ri=stubs_[i].r();
      double zi=stubs_[i].z();
      double xi=-0.5*ri/sqrt(1.0-0.25*ri*ri*rinv_*rinv_);
      double phii=stubs_[i].phi();
      double yi=-phi0_+phii+asin(0.5*ri*rinv_);
      double zxi=(2.0/rinv_)*asin(0.5*ri*rinv_);
      double zyi=zi-z0_-(2.0/rinv_)*t_*asin(0.5*ri*rinv_);
      double tmp=(yi-dphi0-xi*drinv);
      double tmpz=(zyi-dz0-zxi*dt);
      //cout << "yi tmp:"<<yi<<" "<<tmp<<endl;
      //cout << "zyi tmpz:"<<zyi<<" "<<tmpz<<endl;
      tmp/=(2.9e-3/ri);
      tmpz/=0.036;
      chisq1_+=tmp*tmp;
      chisq2_+=tmpz*tmpz;
    }
    
    //cout << "Old chisqrz chisqrzfit :"
    //	 << chisq1_ <<" "<<chisq2_<<endl;;


    //cout << "chisq1:"<<chisq1_<<" chisq2:"<<chisq2_<<endl;
    //cout << "rinv="<<rinv_<<" rinvfit="<<rinvfit_<<endl;
    //cout << "phi0="<<phi0_<<" phi0fit="<<phi0fit_<<endl;
    //cout << "t   ="<<t_<<" tfit   ="<<tfit_<<endl;
    //cout << "z0  ="<<z0_<<" z0fit  ="<<z0fit_<<endl;
    //cout << "pt  ="<<pt()<<"  ptseed="<<ptseed()<<endl;

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

    int simtrackid;
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
  double ipt(double bfield) const { return 0.00299792*bfield/irinvfit(); }
  double ptseed(double bfield) const { return 0.00299792*bfield/rinv_; }

  double phi0() const { return phi0fit_;}
  double iphi0() const { return iphi0fit();}
  double phi0seed() const { return phi0_;}

  double eta() const { static double two_pi=8*atan(1.0);
    return -log(tan(0.5*(0.25*two_pi-atan(tfit_)))); }
  double ieta() const { static double two_pi=8*atan(1.0);
    return -log(tan(0.5*(0.25*two_pi-atan(itfit())))); }
  double etaseed() const { static double two_pi=8*atan(1.0);
    return -log(tan(0.5*(0.25*two_pi-atan(t_)))); }

  double z0() const { return z0fit_; }
  double iz0() const { return iz0fit(); }
  double z0seed() const { return z0_; }

  double chisq1() const {return chisq1_;}
  double chisq2() const {return chisq2_;}

  double chisq1dof() const {return chisq1_/(stubs_.size()-2);}
  double chisq2dof() const {return chisq2_/(stubs_.size()-2);}
  
  double chisq() const {return chisq1_+chisq2_; }
  double chisqdof() const {return (chisq1_+chisq2_)/(2*stubs_.size()-4); }

  double ichisq1() const {return ichisq1_*DX*DX*BASE18*BASE18/(sigmax*sigmax*BASE*BASE);}
  double ichisq2() const {return ichisq2_*DZ*DZ/(sigmaz*sigmaz);}
  double ichisq1dof() const {return ichisq1()/(stubs_.size()-2);}
  double ichisq2dof() const {return ichisq2()/(stubs_.size()-2);}
  double ichisqdof() const {return (ichisq1()+ichisq2())/(2*stubs_.size()-4); }


  double irinvfit() const {return irinvfit_*DX/BASE3;}
  double iphi0fit() const {return iphi0fit_/(1.0*BASE)+seed_.sectorCenter();}
  double itfit() const {return itfit_*DZ/(DX*BASE3);}
  double iz0fit() const {return iz0fit_*DZ;}


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

  double Drphi_[2][20];
  double Drz_[2][20];
  
  double Mrphi_[3][6];
  double Mrz_[3][6];
  
  double MinvDtrphi_[2][20];
  double MinvDtrz_[2][20];

  L1TWord iDrphi_[2][20];
  L1TWord iDrz_[2][20];

  L1TWord iMinvDtrphi_[2][20];
  L1TWord iMinvDtrz_[2][20];


};


#endif
