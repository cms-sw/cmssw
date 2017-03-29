#ifndef FPGATRACKDERTABLE_H
#define FPGATRACKDERTABLE_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
#include "FPGATrackDer.hh"

using namespace std;

class FPGATrackDerTable{

public:

  FPGATrackDerTable() {

    Nlay_=6;
    Ndisk_=5;

    LayerMemBits_=5;
    DiskMemBits_=7;
    
    LayerDiskMemBits_=15;

    alphaBits_=3;

    nextLayerValue_=0;
    nextDiskValue_=0;
    nextLayerDiskValue_=0;
    lastMultiplicity_=(1<<(3*alphaBits_));


    for(int i=0;i<(1<<Nlay_);i++){
      LayerMem_.push_back(-1);
    }

    for(int i=0;i<(1<<(2*Ndisk_));i++){
      DiskMem_.push_back(-1);
    }

    for(int i=0;i<(1<<(LayerMemBits_+DiskMemBits_));i++){
      LayerDiskMem_.push_back(-1);
    }
   
  }

  ~FPGATrackDerTable() {

  }

  FPGATrackDer* getDerivatives(int index){
    return &derivatives_[index];
  }

  FPGATrackDer* getDerivatives(unsigned int layermask, 
			       unsigned int diskmask,
			       unsigned int alphaindex){
    int index=getIndex(layermask,diskmask);
    //if (index<0||index!=17984||alphaindex!=20) {
    if (index<0) {
      return 0;
    }
    //cout << "getDerivatives index alphaindex "<<index<<" "<<alphaindex<<endl;
    return &derivatives_[index+alphaindex];
  }


  int getIndex(unsigned int layermask,unsigned int diskmask) {

    assert(layermask<LayerMem_.size());

    assert(diskmask<DiskMem_.size());

    int layercode=LayerMem_[layermask];
    int diskcode=DiskMem_[diskmask];

    if (diskcode<0||layercode<0) {
      cout << "layermask diskmask : "<<layermask<<" "<<diskmask<<endl;
      return -1;
    }

    assert(layercode>=0);
    assert(layercode<(1<<LayerMemBits_));
    assert(diskcode>=0);
    assert(diskcode<(1<<DiskMemBits_));

    int layerdiskaddress=layercode+(diskcode<<LayerMemBits_);

    assert(layerdiskaddress>=0);
    assert(layerdiskaddress<(1<<(LayerMemBits_+DiskMemBits_)));

    int address=LayerDiskMem_[layerdiskaddress];

    if (address<0) {
      cout << "layermask diskmask : "<<layermask<<" "<<diskmask<<endl;
      return -1;
    }

    assert(address>=0);
    //cout << "address LayerDiskMemBits_ : "<<address<<" "<<LayerDiskMemBits_<<endl;
    assert(address<(1<<LayerDiskMemBits_));

    return address;

  }

  void addEntry(unsigned int layermask, unsigned int diskmask, int multiplicity){

    assert(multiplicity<=(1<<(3*alphaBits_)));

    assert(layermask<(unsigned int)(1<<Nlay_));

    assert(diskmask<(unsigned int)(1<<(2*Ndisk_)));

    if (LayerMem_[layermask]==-1) {
      LayerMem_[layermask]=nextLayerValue_++;
    }
    if (DiskMem_[diskmask]==-1) {
      DiskMem_[diskmask]=nextDiskValue_++;
    }

    int layercode=LayerMem_[layermask];
    int diskcode=DiskMem_[diskmask];

    assert(layercode>=0);
    assert(layercode<(1<<LayerMemBits_));
    assert(diskcode>=0);
    assert(diskcode<(1<<DiskMemBits_));

    int layerdiskaddress=layercode+(diskcode<<LayerMemBits_);

    assert(layerdiskaddress>=0);
    assert(layerdiskaddress<(1<<(LayerMemBits_+DiskMemBits_)));

    int address=LayerDiskMem_[layerdiskaddress];

    if (address!=-1) {
      cout << "Duplicate entry:  layermask="
	   <<layermask<<" diskmaks="<<diskmask<<endl;
    }

    assert(address==-1);  //Should not already have this one!

    LayerDiskMem_[layerdiskaddress]=nextLayerDiskValue_;

    nextLayerDiskValue_+=multiplicity;

    lastMultiplicity_=multiplicity;

    for(int i=0;i<multiplicity;i++) {
      FPGATrackDer tmp;
      tmp.setIndex(layermask,diskmask,i);
      derivatives_.push_back(tmp);
    }

  }

  void readPatternFile(std::string fileName){

    ifstream in(fileName.c_str());
    cout<<"reading fit pattern file "<<fileName<<"\n";
    cout<<"  flags (good/eof/fail/bad): "<<in.good()<<" "<<in.eof()<<" "<<in.fail()<<" "<<in.bad()<<"\n"; 

    while (in.good()) {

      std::string layerstr,diskstr;
      int multiplicity;

      in >>layerstr>>diskstr>>multiplicity;

      if (!in.good()) continue;
      
      char** tmpptr=0;

      int layers=strtol(layerstr.c_str(), tmpptr, 2); 
      int disks=strtol(diskstr.c_str(), tmpptr, 2); 

      //cout << "adding: "<<layers<<" "<<disks<<" "<<multiplicity<<endl;
   
      addEntry(layers,disks,multiplicity);

    }

  }

  int getEntries() const {
    return nextLayerDiskValue_;
  }

  void fillTable() {

    int nentries=getEntries();

    for (int i=0;i<nentries;i++){
      FPGATrackDer& der=derivatives_[i];
      int layermask=der.getLayerMask();
      int diskmask=der.getDiskMask();
      int alphamask=der.getAlphaMask();

      bool print=getIndex(layermask,diskmask)==17984&&alphamask==20;
      print=false;

      if (print) {
	cout << "i "<<i<<" "<<layermask<<" "<<diskmask<<" "
	     <<alphamask<<" "<<print<<endl;
      }

      int nlayers=0;
      //int layers[6];
      double r[6];

      for (unsigned l=0;l<6;l++){
	if (layermask&(1<<(5-l))) {
	  //layers[nlayers]=l+1;
	  r[nlayers]=rmean[l];
	  //cout << "Hit in layer "<<layers[nlayers]<<" "<<r[nlayers]<<endl;
	  nlayers++;  
	}
      }

      int ndisks=0;
      //int disks[5];
      double z[5];
      double alpha[5];

      //double t=sinh(1.2);
      //double rinv=-0.0057/2.5;

      double t=gett(diskmask);
      double rinv=0.00000001;
      //cout << "layermask diskmask t :"<<layermask<<" "<<diskmask<<" "<<t<<endl;
      
      for (unsigned d=0;d<5;d++){
	if (diskmask&(3<<(2*(4-d)))) {
	  //disks[ndisks]=d+1;
	  z[ndisks]=zmean[d];
	  alpha[ndisks]=0.0;
	  if (diskmask&(1<<(2*(4-d)))) {
	    int ialpha=alphamask&7;
	    alphamask=alphamask>>3;
	    double r=zmean[d]/t;
	    alpha[ndisks]=480*0.009*(ialpha-3.5)/(4.0*r*r);
	    //if (d==0) alpha[ndisks]=-0.000220;
	    //if (d==1) alpha[ndisks]=0.000244;
	    if (print) {
	      cout << "Hit in disk "<<z[ndisks]<<" "
		   <<ialpha<<" "<<alpha[ndisks]<<endl;
	    }
	  }
	  ndisks++;  
	}
      }


      double D[4][12];
      double MinvDt[4][12];
      int iMinvDt[4][12];
      

      if (print) {
	for(int ii=0;ii<nlayers;ii++){
	  cout << "Layer r : "<<r[ii]<<endl;
	}
	for(int ii=0;ii<ndisks;ii++){
	  cout << "Disk z alpha : "<<z[ii]<<" "<<alpha[ii]<<endl;
	}
      }

      calculateDerivativesTable(nlayers,r,ndisks,z,alpha,t,rinv,D,MinvDt,iMinvDt);


      for(int j=0;j<nlayers+ndisks;j++){
	if (print) {
	  cout << "Table "<<endl;
	  cout << MinvDt[0][2*j] <<" "
	       << MinvDt[1][2*j] <<" "
	       << MinvDt[2][2*j] <<" "
	       << MinvDt[3][2*j] <<" "
	       <<endl;
	  cout << MinvDt[0][2*j+1] <<" "
	       << MinvDt[1][2*j+1] <<" "
	       << MinvDt[2][2*j+1] <<" "
	       << MinvDt[3][2*j+1] <<" "
	       <<endl;
	}
	
	der.sett(t);

	//integer
	der.setirinvdphi(j,iMinvDt[0][2*j]); 
	der.setirinvdzordr(j,iMinvDt[0][2*j+1]); 
	der.setiphi0dphi(j,iMinvDt[1][2*j]); 
	der.setiphi0dzordr(j,iMinvDt[1][2*j+1]); 
	der.setitdphi(j,iMinvDt[2][2*j]); 
	der.setitdzordr(j,iMinvDt[2][2*j+1]); 
	der.setiz0dphi(j,iMinvDt[3][2*j]); 
	der.setiz0dzordr(j,iMinvDt[3][2*j+1]); 
	//floating point
	der.setrinvdphi(j,MinvDt[0][2*j]); 
	der.setrinvdzordr(j,MinvDt[0][2*j+1]); 
	der.setphi0dphi(j,MinvDt[1][2*j]); 
	der.setphi0dzordr(j,MinvDt[1][2*j+1]); 
	der.settdphi(j,MinvDt[2][2*j]); 
	der.settdzordr(j,MinvDt[2][2*j+1]); 
	der.setz0dphi(j,MinvDt[3][2*j]); 
	der.setz0dzordr(j,MinvDt[3][2*j+1]); 
      }



    }

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



  void calculateDerivativesTable(unsigned int nlayers,
				 double r[6],
				 unsigned int ndisks,
				 double z[5],
				 double alpha[5],
				 double t,
				 double rinv,
				 double D[4][12],
				 double MinvDt[4][12],
				 int iMinvDt[4][12]){


    double sigmax=0.01/sqrt(12.0);
    double sigmaz=0.15/sqrt(12.0);
    double sigmaz2=5.0/sqrt(12.0);

    unsigned int n=nlayers+ndisks;
    
    assert(n<=6);

    double rnew[6];

    int j=0;

    double M[4][8];


    //here we handle a barrel hit
    for(unsigned int i=0;i<nlayers;i++) {

      double ri=r[i];

      rnew[i]=ri;

      //double rinv=0.000001; //should simplify?

      //first we have the phi position
      D[0][j]=-0.5*ri*ri/sqrt(1-0.25*ri*ri*rinv*rinv)/sigmax;
      D[1][j]=ri/sigmax;
      D[2][j]=0.0;
      D[3][j]=0.0;
      j++;
      //second the z position
      D[0][j]=0.0;
      D[1][j]=0.0;
      if (ri<60.0) {
	D[2][j]=(2/rinv)*asin(0.5*ri*rinv)/sigmaz;
	D[3][j]=1.0/sigmaz;
      } else {
	D[2][j]=(2/rinv)*asin(0.5*ri*rinv)/sigmaz2;
	D[3][j]=1.0/sigmaz2;
      }

      //cout << "0 D "<<ri<<" "<<rinv<<" "
      //   <<D[0][j]<<" "<<D[1][j]<<" "<<D[2][j]<<" "<<D[3][j]<<endl;

      j++;

    }


    for(unsigned int i=0;i<ndisks;i++) {

      double zi=z[i];


      //double rinv=0.000001;
      double z0=0.0;
 
      double rmultiplier=alpha[i]*zi/t;
      //rmultiplier=0.0;
      double phimultiplier=zi/t;
      
      //cout << "i zi t : "<<i<<" "<<zi<<" "<<t<<endl;

      double drdrinv=-2.0*sin(0.5*rinv*(zi-z0)/t)/(rinv*rinv)
      +(zi-z0)*cos(0.5*rinv*(zi-z0)/t)/(rinv*t);
      double drdphi0=0;
      double drdt=-(zi-z0)*cos(0.5*rinv*(zi-z0)/t)/(t*t);
      double drdz0=-cos(0.5*rinv*(zi-z0)/t)/t;


      double dphidrinv=-0.5*(zi-z0)/t;
      double dphidphi0=1.0;
      double dphidt=0.5*rinv*(zi-z0)/(t*t);
      double dphidz0=0.5*rinv/t;

      double r=(zi-z0)/t;

      //cout << "r alpha : "<<r<<" "<<alpha[i]<<endl;

      rnew[i+nlayers]=r;

      //cout << "rnew["<<i+nlayers<<"] = "<<rnew[i+nlayers]
      //	   <<" "<<zi<<" "<<z0<<" "<<t<<endl;


      //cout << "FITLINNEW r = "<<r<<endl; 

      //second the rphi position
      double sigmaxtmp=sigmax;
      if (fabs(alpha[i])>1e-10) {
	//cout << "Table for outer disks : "<<r<<" "<<zi<<endl;
	sigmaxtmp*=errfac;
      }

      D[0][j]=(phimultiplier*dphidrinv+rmultiplier*drdrinv)/sigmaxtmp;
      D[1][j]=(phimultiplier*dphidphi0+rmultiplier*drdphi0)/sigmaxtmp;
      D[2][j]=(phimultiplier*dphidt+rmultiplier*drdt)/sigmaxtmp;
      D[3][j]=(phimultiplier*dphidz0+rmultiplier*drdz0)/sigmaxtmp;

      //cout << "1 D "<<D[0][j]<<" "<<D[1][j]<<" "<<D[2][j]<<" "<<D[3][j]<<endl;

      j++;

      //cout << "alpha i r = "<<alpha[i]<<" "<<i<<" "<<r<<endl;
      if (fabs(alpha[i])<1e-10) {
	D[0][j]=drdrinv/sigmaz;
	D[1][j]=drdphi0/sigmaz;
	D[2][j]=drdt/sigmaz;
	D[3][j]=drdz0/sigmaz;
      }
      else {
	D[0][j]=drdrinv/sigmaz2;
	D[1][j]=drdphi0/sigmaz2;
	D[2][j]=drdt/sigmaz2;
	D[3][j]=drdz0/sigmaz2;
      }

      //cout << "2 D "<<D[0][j]<<" "<<D[1][j]<<" "<<D[2][j]<<" "<<D[3][j]<<endl;

      j++;
      

    }

    //cout << "j n : "<<j<<" "<<n<<endl;
    
    for(unsigned int i1=0;i1<4;i1++){
      for(unsigned int i2=0;i2<4;i2++){
	M[i1][i2]=0.0;
	for(unsigned int j=0;j<2*n;j++){
	  M[i1][i2]+=D[i1][j]*D[i2][j];	  
	}
      }
    }

    invert(M,4);

    for(unsigned int j=0;j<12;j++) {
      for(unsigned int i1=0;i1<4;i1++) {
	MinvDt[i1][j]=0.0;
	iMinvDt[i1][j]=0;
      }
    }  

    for(unsigned int j=0;j<2*n;j++) {
      for(unsigned int i1=0;i1<4;i1++) {
	for(unsigned int i2=0;i2<4;i2++) {
	  MinvDt[i1][j]+=M[i1][i2+4]*D[i2][j];
	  //cout << "0 M D = "<<M[i1][i2+4]<<" "<<D[i2][j]<<endl;
	
	}
      }
    }


    for (unsigned int i=0;i<n;i++) {


      //First the barrel
      if (i<nlayers) {

	MinvDt[0][2*i]*=rnew[i]/sigmax;
	MinvDt[1][2*i]*=rnew[i]/sigmax;
	MinvDt[2][2*i]*=rnew[i]/sigmax;
	MinvDt[3][2*i]*=rnew[i]/sigmax;

	//cout << "1 MinvDt[0][2*i] = "<<MinvDt[0][2*i]<<endl;

      
	iMinvDt[0][2*i]=(1<<fitrinvbitshift)*MinvDt[0][2*i]*kphi1/krinvpars;
	iMinvDt[1][2*i]=(1<<fitphi0bitshift)*MinvDt[1][2*i]*kphi1/kphi0pars;
	iMinvDt[2][2*i]=(1<<fittbitshift)*MinvDt[2][2*i]*kphi1/ktpars;
	iMinvDt[3][2*i]=(1<<fitz0bitshift)*MinvDt[3][2*i]*kphi1/kzpars;

	if (rnew[i]<57.0) {
	  MinvDt[0][2*i+1]/=sigmaz;
	  MinvDt[1][2*i+1]/=sigmaz;
	  MinvDt[2][2*i+1]/=sigmaz;
	  MinvDt[3][2*i+1]/=sigmaz;

	  iMinvDt[0][2*i+1]=(1<<fitrinvbitshift)*MinvDt[0][2*i+1]*kzproj/krinvpars;
	  iMinvDt[1][2*i+1]=(1<<fitphi0bitshift)*MinvDt[1][2*i+1]*kzproj/kphi0pars;
	  iMinvDt[2][2*i+1]=(1<<fittbitshift)*MinvDt[2][2*i+1]*kzproj/ktpars;
	  iMinvDt[3][2*i+1]=(1<<fitz0bitshift)*MinvDt[3][2*i+1]*kzproj/kzpars;
	} else {
	  MinvDt[0][2*i+1]/=sigmaz2;
	  MinvDt[1][2*i+1]/=sigmaz2;
	  MinvDt[2][2*i+1]/=sigmaz2;
	  MinvDt[3][2*i+1]/=sigmaz2;

	  iMinvDt[0][2*i+1]=(1<<fitrinvbitshift)*MinvDt[0][2*i+1]*kzproj/krinvpars;
	  iMinvDt[1][2*i+1]=(1<<fitphi0bitshift)*MinvDt[1][2*i+1]*kzproj/kphi0pars;
	  iMinvDt[2][2*i+1]=(1<<fittbitshift)*MinvDt[2][2*i+1]*kzproj/ktpars;
	  iMinvDt[3][2*i+1]=(1<<fitz0bitshift)*MinvDt[3][2*i+1]*kzproj/kzpars;
	}
      }

      //Secondly the disks
      else {


	if (fabs(alpha[i])<1e-10) {
	  MinvDt[0][2*i]*=(rnew[i]/sigmax);
	  MinvDt[1][2*i]*=(rnew[i]/sigmax);
	  MinvDt[2][2*i]*=(rnew[i]/sigmax);
	  MinvDt[3][2*i]*=(rnew[i]/sigmax);
	} else {
	  MinvDt[0][2*i]*=(rnew[i]/(errfac*sigmax));
	  MinvDt[1][2*i]*=(rnew[i]/(errfac*sigmax));
	  MinvDt[2][2*i]*=(rnew[i]/(errfac*sigmax));
	  MinvDt[3][2*i]*=(rnew[i]/(errfac*sigmax));
	}      

	//cout << "2 MinvDt[0][2*i] = "<<MinvDt[0][2*i]<<endl;

	assert(MinvDt[0][2*i]==MinvDt[0][2*i]);

	iMinvDt[0][2*i]=(1<<fitrinvbitshift)*MinvDt[0][2*i]*kphiprojdisk/krinvparsdisk;
	iMinvDt[1][2*i]=(1<<fitphi0bitshift)*MinvDt[1][2*i]*kphiprojdisk/kphi0parsdisk;
	iMinvDt[2][2*i]=(1<<fittbitshift)*MinvDt[2][2*i]*kphiprojdisk/ktparsdisk;
	iMinvDt[3][2*i]=(1<<fitz0bitshift)*MinvDt[3][2*i]*kphiprojdisk/kzdisk;

	if (alpha[i]==0.0) {
	  MinvDt[0][2*i+1]/=sigmaz;
	  MinvDt[1][2*i+1]/=sigmaz;
	  MinvDt[2][2*i+1]/=sigmaz;
	  MinvDt[3][2*i+1]/=sigmaz;
	} else {
	  MinvDt[0][2*i+1]/=sigmaz2;
	  MinvDt[1][2*i+1]/=sigmaz2;
	  MinvDt[2][2*i+1]/=sigmaz2;
	  MinvDt[3][2*i+1]/=sigmaz2;
	}

	iMinvDt[0][2*i+1]=(1<<fitrinvbitshift)*MinvDt[0][2*i+1]*krprojshiftdisk/krinvparsdisk;
	iMinvDt[1][2*i+1]=(1<<fitphi0bitshift)*MinvDt[1][2*i+1]*krprojshiftdisk/kphi0parsdisk;
	iMinvDt[2][2*i+1]=(1<<fittbitshift)*MinvDt[2][2*i+1]*krprojshiftdisk/ktparsdisk;
	iMinvDt[3][2*i+1]=(1<<fitz0bitshift)*MinvDt[3][2*i+1]*krprojshiftdisk/kzdisk;
      
      }

    }
    

  }
  

  double gett(int diskmask) { //should use layers also..

    if (diskmask==0) return 0.0;

    double tmax=1000.0;
    double tmin=0.0;

    for(int d=1;d<=5;d++) {

      if (diskmask&(1<<(2*(5-d)+1))) { //PS hit
	double dmax=zmean[d-1]/22.0;
	if (dmax>sinh(2.4)) dmax=sinh(2.4);
	double dmin=zmean[d-1]/60.0;
	if (dmax<tmax) tmax=dmax;
	if (dmin>tmin) tmin=dmin;
      } 

      if (diskmask&(1<<(2*(5-d)))) { //2S hit
	double dmax=zmean[d-1]/60.0;
	double dmin=zmean[d-1]/105.0;
	if (dmax<tmax) tmax=dmax;
	if (dmin>tmin) tmin=dmin;	
      } 

    }

    return 0.5*(tmax+tmin);

  }


private:


  vector<int> LayerMem_;
  vector<int> DiskMem_;

  vector<int> LayerDiskMem_;

  unsigned int LayerMemBits_;
  unsigned int DiskMemBits_;
  unsigned int LayerDiskMemBits_;
  unsigned int alphaBits_;

  unsigned int Nlay_;
  unsigned int Ndisk_;

  vector<FPGATrackDer> derivatives_;
  
  int nextLayerValue_;
  int nextDiskValue_;
  int nextLayerDiskValue_;
  int lastMultiplicity_;

};



#endif



