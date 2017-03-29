//This class implementes the track fit
#ifndef FPGAFITTRACK_H
#define FPGAFITTRACK_H

#include "FPGAProcessBase.hh"
#include "FPGATrackDerTable.hh"

using namespace std;

class FPGAFitTrack:public FPGAProcessBase{

public:

  FPGAFitTrack(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
    trackfit_=0;
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="trackout"){
      FPGATrackFit* tmp=dynamic_cast<FPGATrackFit*>(memory);
      assert(tmp!=0);
      trackfit_=tmp;
      return;
    }

    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="tpar1in"||
	input=="tpar2in"||
	input=="tpar3in"||
	input=="tpar4in"||
	input=="tpar5in"||
	input=="tpar6in"||
	input=="tpar7in"||
	input=="tpar8in"){
      FPGATrackletParameters* tmp=dynamic_cast<FPGATrackletParameters*>(memory);
      assert(tmp!=0);
      seedtracklet_.push_back(tmp);
      return;
    }
    if (input=="fullmatch1in1"||
	input=="fullmatch1in2"||
	input=="fullmatch1in3"||
	input=="fullmatch1in4"||
	input=="fullmatch1in5"||
	input=="fullmatch1in6"||
	input=="fullmatch1in7"||
	input=="fullmatch1in8"||
	input=="fullmatch1in9"||
	input=="fullmatch1in10"||
	input=="fullmatch1in11"||
	input=="fullmatch1in12"
	){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      fullmatch1_.push_back(tmp);
      return;
    }
    if (input=="fullmatch2in1"||
	input=="fullmatch2in2"||
	input=="fullmatch2in3"||
	input=="fullmatch2in4"||
	input=="fullmatch2in5"||
	input=="fullmatch2in6"||
	input=="fullmatch2in7"||
	input=="fullmatch2in8"||
	input=="fullmatch2in9"||
	input=="fullmatch2in10"||
	input=="fullmatch2in11"||
	input=="fullmatch2in12"
	){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      fullmatch2_.push_back(tmp);
      return;
    }
    if (input=="fullmatch3in1"||
	input=="fullmatch3in2"||
	input=="fullmatch3in3"||
	input=="fullmatch3in4"||
	input=="fullmatch3in5"||
	input=="fullmatch3in6"||
	input=="fullmatch3in7"||
	input=="fullmatch3in8"||
	input=="fullmatch3in9"||
	input=="fullmatch3in10"||
	input=="fullmatch3in11"||
	input=="fullmatch3in12"
	){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      fullmatch3_.push_back(tmp);
      return;
    }
    if (input=="fullmatch4in1"||
	input=="fullmatch4in2"||
	input=="fullmatch4in3"||
	input=="fullmatch4in4"||
	input=="fullmatch4in5"||
	input=="fullmatch4in6"||
	input=="fullmatch4in7"||
	input=="fullmatch4in8"||
	input=="fullmatch4in9"||
	input=="fullmatch4in10"||
	input=="fullmatch4in11"||
	input=="fullmatch4in12"
	){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      fullmatch4_.push_back(tmp);
      return;
    }
    if (input=="fullmatch5in1"||
	input=="fullmatch5in2"||
	input=="fullmatch5in3"||
	input=="fullmatch5in4"||
	input=="fullmatch5in5"||
	input=="fullmatch5in6"||
	input=="fullmatch5in7"||
	input=="fullmatch5in8"||
	input=="fullmatch5in9"||
	input=="fullmatch5in10"||
	input=="fullmatch5in11"||
	input=="fullmatch5in12"
	){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      fullmatch5_.push_back(tmp);
      return;
    }
    if (input=="fullmatch6in1"||
	input=="fullmatch6in2"||
	input=="fullmatch6in3"||
	input=="fullmatch6in4"||
	input=="fullmatch6in5"||
	input=="fullmatch6in6"
	){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      fullmatch6_.push_back(tmp);
      return;
    }
    cout << "Did not find input : "<<input<<endl;
    assert(0);
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




  void calculateDerivativesNew(unsigned int nlayers,
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
	//cout << "Fit with outer disks : "<<r<<" "<<zi<<endl;
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

	  int fact=(1<<(nbitszprojL123-nbitszprojL456));

	  iMinvDt[0][2*i+1]=(1<<fitrinvbitshift)*MinvDt[0][2*i+1]*fact*kzproj/krinvpars;
	  iMinvDt[1][2*i+1]=(1<<fitphi0bitshift)*MinvDt[1][2*i+1]*fact*kzproj/kphi0pars;
	  iMinvDt[2][2*i+1]=(1<<fittbitshift)*MinvDt[2][2*i+1]*fact*kzproj/ktpars;
	  iMinvDt[3][2*i+1]=(1<<fitz0bitshift)*MinvDt[3][2*i+1]*fact*kzproj/kzpars;
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
  


  void trackFitNew(FPGATracklet* tracklet){

    static FPGATrackDerTable derTable;
  

    //test
    static bool first=true;
    if (first) {
      derTable.readPatternFile(fitpatternfile);
      derTable.fillTable();
      cout << "Number of entries in derivative table: "
	   <<derTable.getEntries()<<endl;
      assert(derTable.getEntries()!=0);



      //testDer();
      first=false;
    }

    //cout << "In trackFitNew"<<endl;
 
    //First step is to build list of layers and disks.

    int layers[6];
    double r[6];
    unsigned int nlayers=0;
    int disks[5];
    double z[5];
    unsigned int ndisks=0;


    //int layers[10];
    double phiresid[10];
    double zresid[10];
    double phiresidexact[10];
    double zresidexact[10];
    int iphiresid[10];
    int izresid[10];
    double alpha[10];

    for(unsigned int i=0;i<10;i++){
      iphiresid[i]=0;
      izresid[i]=0;
      alpha[i]=0.0;

      phiresid[i]=0.0;
      zresid[i]=0.0;
      phiresidexact[i]=0.0;
      zresidexact[i]=0.0;
      iphiresid[i]=0;
      izresid[i]=0;
    }


    static ofstream out2;
    if (writeHitPattern) out2.open("hitpattern.txt");

    char matches[8]="000000\0";
    char matches2[12]="0000000000\0";
    int mult=1;


    unsigned int layermask=0;
    unsigned int diskmask=0;
    unsigned int alphaindex=0;
    unsigned int power=1;

    double t=tracklet->t();
    double rinv=tracklet->rinv();

    if (tracklet->isBarrel()) {

      //cout << "Barrel seed"<<endl;

      //nlayers=2;
      //layers[0]=tracklet->layer();
      //layers[1]=tracklet->layer()+1;



      for (unsigned int l=1;l<=6;l++) {
	if (l==(unsigned int)tracklet->layer()||
	    l==(unsigned int)tracklet->layer()+1) {
	  matches[l-1]='1';
	  //cout << "Hit in layer "<<l<<endl;
	  layermask|=(1<<(6-l));
          layers[nlayers++]=l;
	  continue;
	}
	if (tracklet->match(l)) {
	  matches[l-1]='1';
	  //cout << "Hit in layer "<<l<<endl;
	  layermask|=(1<<(6-l));
	  phiresid[nlayers]=tracklet->phiresidapprox(l);
	  zresid[nlayers]=tracklet->zresidapprox(l);
	  phiresidexact[nlayers]=tracklet->phiresid(l);
	  zresidexact[nlayers]=tracklet->zresid(l);
	  iphiresid[nlayers]=tracklet->fpgaphiresid(l).value();
	  izresid[nlayers]=tracklet->fpgazresid(l).value();
	  
	  layers[nlayers++]=l;
	}	
      }



      for (unsigned int d=1;d<=5;d++) {
        if (mult==512) continue;
	//cout << "d ndisks nlayers : "<<d<<" "<<ndisks<<" "<<nlayers<<endl;
	if (ndisks+nlayers>=6) continue;
	if (tracklet->matchdisk(d)) {
	  //cout << "d="<<d<<" t="<<tracklet->t()<<" "<<tracklet->fpgat().value()*ktpars<<endl;
	  if (fabs(tracklet->alphadisk(d))<1e-20) {
	    matches2[2*(5-d)]='1';
	    diskmask|=(1<<(2*(5-d)+1));
	  }
	  else{
	    double alphastub=tracklet->alphadisk(d);
            double rstub=zmean[d-1]/tracklet->t();
	    double alphamax=480*0.009/(rstub*rstub);
            int ialpha=4*(1.0+alphastub/alphamax);
	    if (ialpha<0) ialpha=0;
	    if (ialpha>7) ialpha=7;
	    //if (t<0) ialpha=7-ialpha;
	    //cout << "z alpha alphamax ialpha : "
	    //	 <<zmean[d-1]<<" "<<alphastub<<" "
	    //	 <<alphamax<<" "<<ialpha<<endl;
	    alphaindex+=ialpha*power;
	    power*=8;
	    matches2[2*(d-1)+1]='1';
	    diskmask|=(1<<(2*(5-d)));
	    mult*=8;
	  }
	  alpha[ndisks]=tracklet->alphadisk(d);
	  phiresid[nlayers+ndisks]=tracklet->phiresidapproxdisk(d);
	  zresid[nlayers+ndisks]=tracklet->rresidapproxdisk(d);
	  phiresidexact[nlayers+ndisks]=tracklet->phiresiddisk(d);
	  zresidexact[nlayers+ndisks]=tracklet->rresiddisk(d);
	  iphiresid[nlayers+ndisks]=tracklet->fpgaphiresiddisk(d).value();
	  izresid[nlayers+ndisks]=tracklet->fpgarresiddisk(d).value();
	  
	  disks[ndisks++]=d;
	}
      }

      if (mult<=512) {
	if (writeHitPattern) {
	  out2<<matches<<" "<<matches2<<" "<<mult<<endl;
	}
      }

    } 
    
    if (tracklet->isDisk()) {

      //cout << "Disk seed"<<endl;

      for (unsigned int l=1;l<=2;l++) {
	if (tracklet->match(l)) {
	  matches[l-1]='1';
	  //cout << "Hit in layer "<<l<<endl;
	  layermask|=(1<<(6-l));

	  phiresid[nlayers]=tracklet->phiresidapprox(l);
	  zresid[nlayers]=tracklet->zresidapprox(l);
	  phiresidexact[nlayers]=tracklet->phiresid(l);
	  zresidexact[nlayers]=tracklet->zresid(l);
	  iphiresid[nlayers]=tracklet->fpgaphiresid(l).value();
	  izresid[nlayers]=tracklet->fpgazresid(l).value();
	  
	  layers[nlayers++]=l;
	}
      }


      for (unsigned int d1=1;d1<=5;d1++) {
	int d=d1;
	if (tracklet->fpgat().value()<0.0) d=-d1;
	if (d==tracklet->disk()||  //All seeds in PS modules
	    d==tracklet->disk2()){
	  matches2[2*(5-d1)]='1';
	  diskmask|=(1<<(2*(5-d1)+1));
	  alpha[ndisks]=0.0;
	  disks[ndisks++]=d;
	  continue;
	}

	if (ndisks+nlayers>=6) continue;	
	if (tracklet->matchdisk(d)) {

	  if (fabs(tracklet->alphadisk(d))<1e-20) {
	    matches2[2*(5-d1)]='1';
	    diskmask|=(1<<(2*(5-d1)+1));
	  }
	  else{
	    double alphastub=tracklet->alphadisk(d);
            double rstub=zmean[d-1]/fabs(tracklet->t());
	    double alphamax=480*0.009/(rstub*rstub);
            int ialpha=4*(1.0+alphastub/alphamax);
	    if (ialpha<0) ialpha=0;
	    if (ialpha>7) ialpha=7;
	    //if (t<0) ialpha=7-ialpha;
	    //cout << "z alpha alphamax ialpha : "
	    //	 <<zmean[d1-1]<<" "<<alphastub<<" "
	    // 	 <<alphamax<<" "<<ialpha<<endl;
	    alphaindex+=ialpha*power;
	    power*=8;
	    matches2[2*(d1-1)+1]='1';
	    diskmask|=(1<<(2*(5-d1)));
	    mult*=8;
	  }
	  
	  alpha[ndisks]=tracklet->alphadisk(d);
	  phiresid[nlayers+ndisks]=tracklet->phiresidapproxdisk(d);
	  zresid[nlayers+ndisks]=tracklet->rresidapproxdisk(d);
	  phiresidexact[nlayers+ndisks]=tracklet->phiresiddisk(d);
	  zresidexact[nlayers+ndisks]=tracklet->rresiddisk(d);
	  iphiresid[nlayers+ndisks]=tracklet->fpgaphiresiddisk(d).value();
	  izresid[nlayers+ndisks]=tracklet->fpgarresiddisk(d).value();

	  //cout << "rresid : "<<tracklet->rresiddisk(d)<<" "
	  //     <<tracklet->rresidapproxdisk(d)<<endl;
	  
	  disks[ndisks++]=d;
	}
      }

    } 

    if (tracklet->isOverlap()) {

      //cout << "Overlap seed "<<tracklet->layer()<<endl;

      //nlayers=1;

      //layers[0]=tracklet->layer();

     
      for (unsigned int l=1;l<=2;l++) {
	if (l==(unsigned int)tracklet->layer()) {
	  matches[l-1]='1';
	  //cout << "Seed hit in layer "<<l<<endl;
	  layermask|=(1<<(6-l));
          layers[nlayers++]=l;
	  continue;
	}
	if (tracklet->match(l)) {
	  matches[l-1]='1';
	  //cout << "Match hit in layer "<<l<<endl;
	  layermask|=(1<<(6-l));
	  phiresid[nlayers]=tracklet->phiresidapprox(l);
	  zresid[nlayers]=tracklet->zresidapprox(l);
	  phiresidexact[nlayers]=tracklet->phiresid(l);
	  zresidexact[nlayers]=tracklet->zresid(l);
	  iphiresid[nlayers]=tracklet->fpgaphiresid(l).value();
	  izresid[nlayers]=tracklet->fpgazresid(l).value();

	  //cout <<  "FITOVERLAP1 "<<phiresid[nlayers]<<" "
	  //     <<iphiresid[nlayers]*kphi1<<endl; 
	  //cout <<  "FITOVERLAP2 "<<zresid[nlayers]<<" "
	  //     <<izresid[nlayers]*kzproj<<endl; 

	  layers[nlayers++]=l;
	}
      }


      //ndisks=1;
      //disks[0]=tracklet->disk();

      //for (unsigned int i=0;i<2;i++) {
      //	alpha[nlayers+i]=0.0;
      //	phiresid[nlayers+i]=0.0;
      //	zresid[nlayers+i]=0.0;
      //	phiresidexact[nlayers+i]=0.0;
      //	zresidexact[nlayers+i]=0.0;
      //	iphiresid[nlayers+i]=0;
      //	izresid[nlayers+i]=0;
      //}



      for (unsigned int d1=1;d1<=5;d1++) {
	//cout << "d1 diskmask : "<<d1<<" "<<diskmask<<endl;
        if (mult==512) continue;
	int d=d1;
	if (tracklet->fpgat().value()<0.0) d=-d1;
	if (d==tracklet->disk()){  //All seeds in PS modules
	  disks[ndisks]=tracklet->disk();
	  matches2[2*(5-d1)]='1';
	  diskmask|=(1<<(2*(5-d1)+1));
	  alpha[ndisks]=0.0;
	  ndisks++;
	  continue;
	}


	if (ndisks+nlayers>=6) continue;	
	if (tracklet->matchdisk(d)) {
	  if (fabs(tracklet->alphadisk(d))<1e-20) {
	    matches2[2*(5-d1)]='1';
	    diskmask|=(1<<(2*(5-d1)+1));
	  }
	  else{
	    double alphastub=tracklet->alphadisk(d);
            double rstub=zmean[d-1]/tracklet->t();
	    double alphamax=480*0.009/(rstub*rstub);
            int ialpha=4*(1.0+alphastub/alphamax);
	    if (ialpha<0) ialpha=0;
	    if (ialpha>7) ialpha=7;
	    //if (t<0) ialpha=7-ialpha;
	    //cout << "z alpha alphamax ialpha : "
	    // 	 <<zmean[d1-1]<<" "<<alphastub<<" "
	    //	 <<alphamax<<" "<<ialpha<<endl;
	    alphaindex+=ialpha*power;
	    power*=8;
	    matches2[2*(d1-1)+1]='1';
	    diskmask|=(1<<(2*(5-d1)));
	    mult*=8;
	  }
	  

	  alpha[ndisks]=tracklet->alphadisk(d);
	  phiresid[nlayers+ndisks]=tracklet->phiresidapproxdisk(d);
	  zresid[nlayers+ndisks]=tracklet->rresidapproxdisk(d);
	  phiresidexact[nlayers+ndisks]=tracklet->phiresiddisk(d);
	  zresidexact[nlayers+ndisks]=tracklet->rresiddisk(d);
	  iphiresid[nlayers+ndisks]=tracklet->fpgaphiresiddisk(d).value();
	  izresid[nlayers+ndisks]=tracklet->fpgarresiddisk(d).value();

	  //cout << "rresid : "<<tracklet->rresiddisk(d)<<" "
	  //     <<tracklet->rresidapproxdisk(d)<<endl;
	  
	  disks[ndisks++]=d;
	}
      }

    } 


    //cout << "layermask diskmask alphaindex: "
    //	 <<layermask<<" "<<diskmask<<" "<<alphaindex<<endl;


    FPGATrackDer* derivatives=derTable.getDerivatives(layermask, diskmask,alphaindex);

    if (derivatives==0) {
      cout << "No derivative for layermask, diskmask : "
	   <<layermask<<" "<<diskmask<<" eta = "<<asinh(t)<<endl;
      return;
    }



    //cout << "ndisks : "<<ndisks<<endl;



    //cout << "t_track t_der : "<<t<<" "<<derivatives.gett()<<endl;


    int sign=1;
    if (t<0.0) sign=-1;

    double rstub[6];


    //cout << "trackFitNew layers: ";
    for (unsigned i=0;i<nlayers;i++){
      r[i]=rmean[layers[i]-1];
      //cout << "i layers[i] r[i] : "<<i<<" "<<layers[i]<<" "<<r[i]<<endl;
      rstub[i]=r[i];
      //cout <<" "<<layers[i];
    }
    //cout << " disks: ";
    for (unsigned i=0;i<ndisks;i++){
      //cout << "i disks[i] = "<<i<<" "<<disks[i]<<endl;
      z[i]=sign*zmean[abs(disks[i])-1];
      rstub[i+nlayers]=z[i]/t;
      //cout << "i disks z alpha = "<<i<<" "<<disks[i]
      //	   <<" "<<z[i]<<" "<<alpha[i]<<endl;
      //cout << "zi = "<<z[i]<<endl;
      //cout <<" "<<disks[i];
    }
    //cout << endl;


    double D[4][12];
    double MinvDt[4][12];
    int iMinvDt[4][12];

    unsigned int n=nlayers+ndisks;
   
    /* 
    for(unsigned int ii=0;ii<nlayers;ii++){
      cout << "Fit Layer r : "<<r[ii]<<endl;
    }
    for(unsigned int ii=0;ii<ndisks;ii++){
      cout << "Fit Disk z alpha : "<<z[ii]<<" "<<alpha[ii]<<endl;
    }
    cout << "t= "<<t<<endl;
    cout << "rinv= "<<rinv<<endl;
    */

    if (exactderivatives) {
      calculateDerivativesNew(nlayers,r,ndisks,z,alpha,t,rinv,D,MinvDt,iMinvDt);

      /*
      cout << "Normal : "<<nlayers<<" "<<ndisks<<endl;
      for (unsigned int ii=0;ii<2*(nlayers+ndisks);ii++){
	cout <<MinvDt[0][ii]<<" "<<MinvDt[1][ii]<<" "
	     <<MinvDt[2][ii]<<" "<<MinvDt[3][ii]<<endl;
      }
      t=-t;
      for (unsigned int ii=0;ii<ndisks;ii++){
	z[ii]=-z[ii];
	alpha[ii]=-alpha[ii];
      }
      calculateDerivativesNew(nlayers,r,ndisks,z,alpha,t,rinv,D,MinvDt,iMinvDt);      cout << "Negative : "<<nlayers<<" "<<ndisks<<endl;
      for (unsigned int ii=0;ii<2*(nlayers+ndisks);ii++){
	cout <<MinvDt[0][ii]<<" "<<MinvDt[1][ii]<<" "
	     <<MinvDt[2][ii]<<" "<<MinvDt[3][ii]<<endl;
      }
      */

    } else {
      if (exactderivativesforfloating) {
	derivatives->fill(tracklet->fpgat().value(),MinvDt,iMinvDt);
	/*
	cout << "Floating Table: "<<nlayers<<" "<<ndisks<<endl;
	for (unsigned int ii=0;ii<2*(nlayers+ndisks);ii++){
	  cout <<MinvDt[0][ii]<<" "<<MinvDt[1][ii]<<" "
	       <<MinvDt[2][ii]<<" "<<MinvDt[3][ii]<<endl;
	}
	*/
	int iMinvDtDummy[4][12];
	calculateDerivativesNew(nlayers,r,ndisks,z,alpha,t,rinv,
				D,MinvDt,iMinvDtDummy);
      } else {
	derivatives->fill(tracklet->fpgat().value(),MinvDt,iMinvDt);
      }
    }
    
    /*
    cout << "Floating : "<<nlayers<<" "<<ndisks<<endl;
    for (unsigned int ii=0;ii<2*(nlayers+ndisks);ii++){
      cout <<MinvDt[0][ii]<<" "<<MinvDt[1][ii]<<" "
	   <<MinvDt[2][ii]<<" "<<MinvDt[3][ii]<<endl;
    }

    cout << "Integer : "<<nlayers<<" "<<ndisks<<endl;
    for (unsigned int ii=0;ii<2*(nlayers+ndisks);ii++){
      cout <<iMinvDt[0][ii]<<" "<<iMinvDt[1][ii]<<" "
	   <<iMinvDt[2][ii]<<" "<<iMinvDt[3][ii]<<endl;
    }
    */


    double rinvseed=tracklet->rinvapprox();
    double phi0seed=tracklet->phi0approx();
    double tseed=tracklet->tapprox();
    double z0seed=tracklet->z0approx();

    double rinvseedexact=tracklet->rinv();
    double phi0seedexact=tracklet->phi0();
    double tseedexact=tracklet->t();
    double z0seedexact=tracklet->z0();



    double chisqseed=0.0;
    double chisqseedexact=0.0;

    double delta[12];
    double deltaexact[12];
    int idelta[12];

    for(unsigned int i=0;i<12;i++) {
      delta[i]=0.0;
      deltaexact[i]=0.0;
      idelta[i]=0;
    }

    int j=0;

    for(unsigned int i=0;i<n;i++) {

      idelta[j]=iphiresid[i];
      delta[j]=phiresid[i];
      deltaexact[j++]=phiresidexact[i];

      idelta[j]=izresid[i];
      delta[j]=zresid[i];
      deltaexact[j++]=zresidexact[i];
 
      chisqseed+=(delta[j-2]*delta[j-2]+delta[j-1]*delta[j-1]); 
      chisqseedexact+=(deltaexact[j-2]*deltaexact[j-2]+
		       deltaexact[j-1]*deltaexact[j-1]);
    }
    assert(j<=12);
    
    double drinv=0.0;
    double dphi0=0.0;
    double dt=0.0;
    double dz0=0.0;

    double drinvexact=0.0;
    double dphi0exact=0.0;
    double dtexact=0.0;
    double dz0exact=0.0;

    int idrinv=0;
    int idphi0=0;
    int idt=0;
    int idz0=0;


    double drinv_cov=0.0;
    double dphi0_cov=0.0;
    double dt_cov=0.0;
    double dz0_cov=0.0;

    double drinv_covexact=0.0;
    double dphi0_covexact=0.0;
    double dt_covexact=0.0;
    double dz0_covexact=0.0;



    for(unsigned int j=0;j<2*n;j++) {

      //cout << "j deltaj dt : "<<j<<" "<<delta[j]<<" "
      //	   <<MinvDt[2][j]*delta[j]<<endl;

      //if (j%2==0) {
      //	cout << "j MinvDt[0][j] getrinvdphi(j/2) :"
      //	     <<j<<" "<<MinvDt[0][j]<<" "<<derivatives.getrinvdphi(j/2)<<endl;
      //      }

      drinv-=MinvDt[0][j]*delta[j];
      dphi0-=MinvDt[1][j]*delta[j];
      dt-=MinvDt[2][j]*delta[j];
      dz0-=MinvDt[3][j]*delta[j];



      drinv_cov+=D[0][j]*delta[j];
      dphi0_cov+=D[1][j]*delta[j];
      dt_cov+=D[2][j]*delta[j];
      dz0_cov+=D[3][j]*delta[j];


      drinvexact-=MinvDt[0][j]*deltaexact[j];
      dphi0exact-=MinvDt[1][j]*deltaexact[j];
      dtexact-=MinvDt[2][j]*deltaexact[j];
      dz0exact-=MinvDt[3][j]*deltaexact[j];

      //cout << "j = "<<j<<" dt, dtexact = "<<MinvDt[2][j]*delta[j]<<" "
      //   <<MinvDt[2][j]*deltaexact[j]<<" "<<delta[j]<<" "
      //   <<deltaexact[j]<<endl;

      drinv_covexact+=D[0][j]*deltaexact[j];
      dphi0_covexact+=D[1][j]*deltaexact[j];
      dt_covexact+=D[2][j]*deltaexact[j];
      dz0_covexact+=D[3][j]*deltaexact[j];

      idrinv-=((iMinvDt[0][j]*idelta[j]));
      idphi0-=((iMinvDt[1][j]*idelta[j]));
      idt-=((iMinvDt[2][j]*idelta[j]));
      idz0-=((iMinvDt[3][j]*idelta[j]));

      /*

      double factor=1024*krprojshiftdisk/ktparsdisk;
      if (j%2==0) {
	factor=1024*kphiprojdisk/ktparsdisk;
      }

      cout << "j dt idt : "
	   <<j<<" "
	   <<MinvDt[2][j]*delta[j]<<" "
	   <<iMinvDt[2][j]*idelta[j]*ktpars/1024.0<<" "
	   <<MinvDt[2][j]<<" "
	   <<iMinvDt[2][j]/factor
	   <<endl;
      */

      if (0&&j%2==0) {

	cout << "DUMPFITLINNEW1"<<" "<<j
	     <<" "<<rinvseed
	     <<" + "<<MinvDt[0][j]*delta[j]
	     <<" "<<MinvDt[0][j]
	     <<" "<<delta[j]*rstub[j/2]*10000
	     <<endl;

	cout << "DUMPFITLINNEW2"<<" "<<j
	     <<" "<<tracklet->fpgarinv().value()*krinvpars
	     <<" + "<<((iMinvDt[0][j]*idelta[j]))*krinvpars/1024.0
	     <<" "<<iMinvDt[0][j]*krinvparsdisk/kphiprojdisk/1024.0
	     <<" "<<idelta[j]*kphiproj123*rstub[j/2]*10000
	     <<" "<<idelta[j]
	     <<endl;

      }


    }
    
    double deltaChisq=drinv*drinv_cov+dphi0*dphi0_cov+dt*dt_cov+dz0*dz0_cov;

    double deltaChisqexact=drinvexact*drinv_covexact+
      dphi0exact*dphi0_covexact+
      dtexact*dt_covexact+
      dz0exact*dz0_covexact;


    int irinvseed=tracklet->fpgarinv().value();
    int iphi0seed=tracklet->fpgaphi0().value();

    int itseed=tracklet->fpgat().value();
    int iz0seed=tracklet->fpgaz0().value();

    int irinvfit=irinvseed-(idrinv>>fitrinvbitshift);
    int iphi0fit=iphi0seed-(idphi0>>fitphi0bitshift);

    int itfit=itseed-(idt>>fittbitshift);
    int iz0fit=iz0seed-(idz0>>fitz0bitshift);

    int ichisqfit=0;

    double rinvfit=rinvseed-drinv;
    double phi0fit=phi0seed-dphi0;

    //cout << "deltaj : "<<tseed<<" "<<dt<<endl;

    //cout << "dt dtexact : "<<dt<<" "<<dtexact<<endl;

    double tfit=tseed-dt;
    double z0fit=z0seed-dz0;

    double chisqfit=chisqseed+deltaChisq;

    double rinvfitexact=rinvseedexact-drinvexact;
    double phi0fitexact=phi0seedexact-dphi0exact;

    double tfitexact=tseedexact-dtexact;
    double z0fitexact=z0seedexact-dz0exact;

    double chisqfitexact=chisqseedexact+deltaChisqexact;


    /*
    cout << "rinvfit  : "<<rinvseed<<" "<<rinvseedexact<<" "
	 <<irinvseed*krinvpars<<"   "
	 <<rinvfit<<" "<<rinvfitexact<<" "<<irinvfit*krinvpars<<endl
	 << "phi0fit  : "<<phi0seed<<" "<<phi0seedexact
	 <<" "<<iphi0seed*kphi0pars<<"   "
	 <<phi0fit<<" "<<phi0fitexact<<" "<<iphi0fit*kphi0pars<<endl
	 << "tfit     : "<<tseed<<" "<<tseedexact<<" "<<itseed*ktpars<<"   "
	 <<tfit<<" "<<tfitexact<<" "<<itfit*ktpars<<endl
	 << "z0fit    : "<<z0seed<<" "<<z0seedexact<<" "
	 <<iz0seed*kzpars<<"   "
	 <<z0fit<<" "<<z0fitexact<<" "<<iz0fit*kzpars<<endl
	 << "chisq fit:"<<chisqseed<<" "<<chisqseedexact<<" "
	 <<chisqfit<<" "<<chisqfitexact<<" "<<ichisqfit
	 << endl;
    */

    //cout << "LINNEW adding new trackfit nmatchesDisk = "
    //	 <<tracklet->nMatchesDisk()<<endl;

    tracklet->setFitPars(rinvfit,phi0fit,tfit,z0fit,chisqfit,
			 rinvfitexact,phi0fitexact,tfitexact,
			 z0fitexact,chisqfitexact,
			 irinvfit,iphi0fit,itfit,iz0fit,ichisqfit);


  }



  void execute(std::vector<FPGATrack>& tracks) {

    //cout << "Fit track in "<<getName()<<endl;


    //Again taking a bit of a short cut in order to reuse old code
    map<FPGATracklet*,int> counts;


    for(unsigned int j=0;j<fullmatch1_.size();j++){
      for(unsigned int i=0;i<fullmatch1_[j]->nMatches();i++){
	FPGATracklet* tracklet=fullmatch1_[j]->getFPGATracklet(i);
	//cout << "In FPGAFitTrack 1 "<<tracklet<<endl;
	if (getName()=="FT_L1L2"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_L3L4"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_L5L6"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_F1F2"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_B1B2"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_F3F4"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_B3B4"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_F1L"&&!tracklet->isOverlap()) continue;
	if (getName()=="FT_B1L"&&!tracklet->isOverlap()) continue;

	if (getName()=="FT_L1L2"&&tracklet->layer()!=1) continue;
	if (getName()=="FT_L3L4"&&tracklet->layer()!=3) continue;
	if (getName()=="FT_L5L6"&&tracklet->layer()!=5) continue;
	if (getName()=="FT_F1F2"&&tracklet->disk()!=1) continue;
	if (getName()=="FT_B1B2"&&tracklet->disk()!=-1) continue;
	if (getName()=="FT_F3F4"&&tracklet->disk()!=3) continue;
	if (getName()=="FT_B3B4"&&tracklet->disk()!=-3) continue;
	if (counts.find(tracklet)==counts.end()){
	  counts[tracklet]=1;
	}
	else{
	  counts[tracklet]++;
	}
      }
    }

    for(unsigned int j=0;j<fullmatch2_.size();j++){
      for(unsigned int i=0;i<fullmatch2_[j]->nMatches();i++){
	FPGATracklet* tracklet=fullmatch2_[j]->getFPGATracklet(i);
	//cout << "In FPGAFitTrack 2 "<<tracklet<<" "<<getName()<<endl;
	if (getName()=="FT_L1L2"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_L3L4"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_L5L6"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_F1F2"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_B1B2"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_F3F4"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_B3B4"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_F1L"&&!tracklet->isOverlap()) continue;
	if (getName()=="FT_B1L"&&!tracklet->isOverlap()) continue;

	if (getName()=="FT_L1L2"&&tracklet->layer()!=1) continue;
	if (getName()=="FT_L3L4"&&tracklet->layer()!=3) continue;
	if (getName()=="FT_L5L6"&&tracklet->layer()!=5) continue;
	if (getName()=="FT_F1F2"&&tracklet->disk()!=1) continue;
	if (getName()=="FT_B1B2"&&tracklet->disk()!=-1) continue;
	if (getName()=="FT_F3F4"&&tracklet->disk()!=3) continue;
	if (getName()=="FT_B3B4"&&tracklet->disk()!=-3) continue;
	//cout << "Found match2 for "<<tracklet<<endl;
	if (counts.find(tracklet)==counts.end()){
	  counts[tracklet]=1;
	}
	else{
	  counts[tracklet]++;
	}
      }
    }

    for(unsigned int j=0;j<fullmatch3_.size();j++){
      for(unsigned int i=0;i<fullmatch3_[j]->nMatches();i++){
	FPGATracklet* tracklet=fullmatch3_[j]->getFPGATracklet(i);
	//cout << "In FPGAFitTrack 3 "<<tracklet<<endl;
	if (getName()=="FT_L1L2"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_L3L4"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_L5L6"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_F1F2"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_B1B2"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_F3F4"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_B3B4"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_F1L"&&!tracklet->isOverlap()) continue;
	if (getName()=="FT_B1L"&&!tracklet->isOverlap()) continue;

	if (getName()=="FT_L1L2"&&tracklet->layer()!=1) continue;
	if (getName()=="FT_L3L4"&&tracklet->layer()!=3) continue;
	if (getName()=="FT_L5L6"&&tracklet->layer()!=5) continue;
	if (getName()=="FT_F1F2"&&tracklet->disk()!=1) continue;
	if (getName()=="FT_B1B2"&&tracklet->disk()!=-1) continue;
	if (getName()=="FT_F3F4"&&tracklet->disk()!=3) continue;
	if (getName()=="FT_B3B4"&&tracklet->disk()!=-3) continue;
	//cout << "Found match3 for "<<tracklet<<endl;
	if (counts.find(tracklet)==counts.end()){
	  counts[tracklet]=1;
	}
	else{
	  counts[tracklet]++;
	}
      }
    }

    for(unsigned int j=0;j<fullmatch4_.size();j++){
      for(unsigned int i=0;i<fullmatch4_[j]->nMatches();i++){
	FPGATracklet* tracklet=fullmatch4_[j]->getFPGATracklet(i);
	//cout << "In FPGAFitTrack 4 "<<tracklet<<endl;
	if (getName()=="FT_L1L2"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_L3L4"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_L5L6"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_F1F2"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_B1B2"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_F3F4"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_B3B4"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_F1L"&&!tracklet->isOverlap()) continue;
	if (getName()=="FT_B1L"&&!tracklet->isOverlap()) continue;

	if (getName()=="FT_L1L2"&&tracklet->layer()!=1) continue;
	if (getName()=="FT_L3L4"&&tracklet->layer()!=3) continue;
	if (getName()=="FT_L5L6"&&tracklet->layer()!=5) continue;
	if (getName()=="FT_F1F2"&&tracklet->disk()!=1) continue;
	if (getName()=="FT_B1B2"&&tracklet->disk()!=-1) continue;
	if (getName()=="FT_F3F4"&&tracklet->disk()!=3) continue;
	if (getName()=="FT_B3B4"&&tracklet->disk()!=-3) continue;
	//cout << "Found match4 for "<<tracklet<<endl;
	if (counts.find(tracklet)==counts.end()){
	  counts[tracklet]=1;
	}
	else{
	  counts[tracklet]++;
	}
      }
    }

    for(unsigned int j=0;j<fullmatch5_.size();j++){
      for(unsigned int i=0;i<fullmatch5_[j]->nMatches();i++){
	FPGATracklet* tracklet=fullmatch5_[j]->getFPGATracklet(i);
	//cout << "In FPGAFitTrack 5 "<<tracklet<<endl;
	if (getName()=="FT_L1L2"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_L3L4"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_L5L6"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_F1F2"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_B1B2"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_F3F4"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_B3B4"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_F1L"&&!tracklet->isOverlap()) continue;
	if (getName()=="FT_B1L"&&!tracklet->isOverlap()) continue;

	if (getName()=="FT_L1L2"&&tracklet->layer()!=1) continue;
	if (getName()=="FT_L3L4"&&tracklet->layer()!=3) continue;
	if (getName()=="FT_L5L6"&&tracklet->layer()!=5) continue;
	if (getName()=="FT_F1F2"&&tracklet->disk()!=1) continue;
	if (getName()=="FT_B1B2"&&tracklet->disk()!=-1) continue;
	if (getName()=="FT_F3F4"&&tracklet->disk()!=3) continue;
	if (getName()=="FT_B3B4"&&tracklet->disk()!=-3) continue;
	//cout << "Found match5 for "<<tracklet<<endl;
	if (counts.find(tracklet)==counts.end()){
	  counts[tracklet]=1;
	}
	else{
	  counts[tracklet]++;
	}
      }
    }

    for(unsigned int j=0;j<fullmatch6_.size();j++){
      for(unsigned int i=0;i<fullmatch6_[j]->nMatches();i++){
	FPGATracklet* tracklet=fullmatch6_[j]->getFPGATracklet(i);
	//cout << "In FPGAFitTrack 6 "<<tracklet<<endl;
	if (getName()=="FT_L1L2"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_L3L4"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_L5L6"&&!tracklet->isBarrel()) continue;
	if (getName()=="FT_F1F2"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_B1B2"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_F3F4"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_B3B4"&&!tracklet->isDisk()) continue;
	if (getName()=="FT_F1L"&&!tracklet->isOverlap()) continue;
	if (getName()=="FT_B1L"&&!tracklet->isOverlap()) continue;

	if (getName()=="FT_L1L2"&&tracklet->layer()!=1) continue;
	if (getName()=="FT_L3L4"&&tracklet->layer()!=3) continue;
	if (getName()=="FT_L5L6"&&tracklet->layer()!=5) continue;
	if (getName()=="FT_F1F2"&&tracklet->disk()!=1) continue;
	if (getName()=="FT_B1B2"&&tracklet->disk()!=-1) continue;
	if (getName()=="FT_F3F4"&&tracklet->disk()!=3) continue;
	if (getName()=="FT_B3B4"&&tracklet->disk()!=-3) continue;
	if (counts.find(tracklet)==counts.end()){
	  counts[tracklet]=1;
	}
	else{
	  counts[tracklet]++;
	}
      }
    }



    for (std::map<FPGATracklet*,int>::iterator it=counts.begin(); it!=counts.end(); ++it){

      FPGATracklet* tracklet=it->first;
      int nmatch=tracklet->nMatches();
      int nmatchdisk=tracklet->nMatchesDisk();

      //cout << getName()<<" "<<tracklet<<" "<<iSector_<<" "<<tracklet->layer()
      //     <<" "<<tracklet->disk()
      //     <<" "<<it->second<<" "<<nmatch<<" "<<nmatchdisk 
      //     << endl;

      bool found=false;

      for(unsigned int i=0;i<seedtracklet_.size();i++){
	for (unsigned int j=0;j<seedtracklet_[i]->nTracklets();j++) {
	  if (tracklet==seedtracklet_[i]->getFPGATracklet(j)) found=true;
	  //cout <<"seed : "<<seedtracklet_[i]->getFPGATracklet(j)<<endl;
	}
      }

      assert(found);

      //if (it->second<2) continue;

      //static ofstream out("nmatches1.txt");
      //	out << tracklet->nMatches()+tracklet->nMatchesDisk() << endl;
      if (nmatch+nmatchdisk>1) {
	//cout << "Will perform trackfit"<<endl;
	trackFitNew(tracklet);
	if (tracklet->fit()){
	  //cout << "Performed track fit in "<<getName()<<endl;
	  assert(trackfit_!=0);
	  trackfit_->addTrack(tracklet);
	  //cout << "Adding track to tracks"<<endl;
	  tracks.push_back(tracklet->getTrack());
	}
      }

    }
    


  }


private:
  
  vector<FPGATrackletParameters*> seedtracklet_;
  vector<FPGAFullMatch*> fullmatch1_;
  vector<FPGAFullMatch*> fullmatch2_;
  vector<FPGAFullMatch*> fullmatch3_;
  vector<FPGAFullMatch*> fullmatch4_;
  vector<FPGAFullMatch*> fullmatch5_;
  vector<FPGAFullMatch*> fullmatch6_;

  FPGATrackFit* trackfit_;

};

#endif
