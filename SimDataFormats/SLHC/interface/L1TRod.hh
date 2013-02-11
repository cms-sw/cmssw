#ifndef L1TROD_H
#define L1TROD_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
using namespace std;

#include "L1TStub.hh"
#include "L1TTracklets.hh"
#include "L1TConstants.hh"
#include "slhcevent.hh"


class L1TRod{

private:
  L1TRod(){
  }


public:

  L1TRod(int sector,int layer,int local_rod){
    sector_=sector;
    layer_=layer;
    local_rod_=local_rod;
    r1_=-999.9;
    r2_=-999.9;
    phi1_=-999.9;
    phi2_=-999.9;
  }

  

  void addStub(const L1TStub& aStub){
    stubs_.push_back(aStub);
    stubmultiplicity_[aStub.module()]++;
    int iz=aStub.iz();
    int iphi=aStub.iphi();
    int nz=iz/16;
    int nphi=iphi/200;
    int iroc=nphi*5+nz;
    //cout << "iroc:"<<iroc<<" "<<iz<<" "<<iphi<<" "<<nz<<" "<<nphi<<endl;
    stubmultiplicityroc_[aStub.module()*25+iroc]++;
  }


  L1TTracklets findTracklets(const L1TRod& aRod) {
            
    L1TTracklets tmp;

    //loop over the pair of rods and look for 
    //stubs.

    //cout << "In L1TRod::findTracklets: "<<stubs_.size()<<" "
    //	 << aRod.stubs_.size()<<endl;

    for(unsigned int i=0;i<stubs_.size();i++){
      double r1=stubs_[i].r();
      double z1=stubs_[i].z();
      double phi1=stubs_[i].phi();

      double phic;
      double x1=getx(r1,phi1);
      double r=getrmin(phic);

      //cout << "phi1:"<<phi1<<endl;

      for(unsigned int j=0;j<aRod.stubs_.size();j++){
	double r2=aRod.stubs_[j].r();
	double z2=aRod.stubs_[j].z();
	double phi2=aRod.stubs_[j].phi();

	double rinv,phi0,z0,t;

        exactTracklet(rinv, phi0, z0, t, r1, phi1, z1, r2, phi2, z2);

	//terminate if not interesting match.
	if (!(fabs(z0)<30.0&&fabs(rinv)<0.0057)) continue;

        double x2=aRod.getx(r2,phi2);
	double Delta=sqrt((this->x1()-aRod.x1())*(this->x1()-aRod.x1())+
			  (this->y1()-aRod.y1())*(this->y1()-aRod.y1()));

	double rinv_n,phi0_n,z0_n,t_n;

	approxTracklet(rinv_n, phi0_n, z0_n, t_n,r, x1, x2, z1, z2, Delta);

	double rinv_b,phi0_b,z0_b,t_b;
	int irinv,iphi0,iz0,it;

	binaryTracklet(rinv_b, phi0_b, z0_b, t_b,
		       irinv, iphi0, iz0, it,
		       r, x1, x2, z1, z2, Delta,phic);


        if (1) {

	  static ofstream out("trackletbinary.txt");
	  out 
	    //<< SLHCEvent::mc_rinv << " "
	    //<< SLHCEvent::mc_phi0 << " "
	    //<< SLHCEvent::mc_z0 << " "
	    //<< SLHCEvent::mc_t << " "
	    << r1 << " " << x1 << " " << z1 << " "
	    << rinv << " " << phi0 << " "<<z0<<" " << t << " "
	    << rinv_n << " " << phi0_n+phic << " "<<z0_n<<" " << t_n << " "
	    << rinv_b << " " << phi0_b+phic << " "<<z0_b<<" " << t_b  
	    << endl;
	}

        if (1) {

	  static ofstream out("tracklet.txt");
	  out << r1 << " " << rinv << " "<<z0<<" "
	      <<stubs_[i].module()<<" "
	      <<aRod.stubs_[j].module()<<endl;
	}

	
	if (fabs(z0)<30.0&&fabs(rinv)<0.0057) {

	  if (fabs(irinv*DX/BASE3-rinv)>0.00007) {
	    //cout << "Warning irinv and rinv do not match:"
	    //	 <<irinv*DX/BASE3<<" "<<rinv<<" will adjust irinv"<<endl;
	    irinv=rinv*BASE3/DX;
	  }

	  //cout << "phi0 iphi0 phic phiSectorCenter_: "
	  //     <<phi0<<" "<<iphi0/(1.0*BASE)<<" "<<phic<<" "<<phiSectorCenter_<<endl;
	    
	  double deltaphi1=phi0-(iphi0/(1.0*BASE)+phiSectorCenter_);
	  if (deltaphi1>0.5*two_pi) deltaphi1-=two_pi;
	  if (deltaphi1<-0.5*two_pi) deltaphi1+=two_pi;
	  if (fabs(deltaphi1)>0.0005) {
	    //cout << "Warning iphi0 and iphi do not match:"
	    //	 <<phi0<<" "<<iphi0/(1.0*BASE)<<" "<<phiSectorCenter_
	    //	 <<" will adjust iphi0"<<endl;
	    iphi0=(phi0-phiSectorCenter_)*BASE;
	  }

	  if (fabs(t-it*DZ/(DX*BASE3))>0.005) {
	    //cout << "Warning t and it do not match"<<endl;
	    it=t*DX*BASE3/DZ;
	  }

	  if (fabs(z0-iz0*DZ)>0.1){
	    //cout << "Warning z0 and iz0 do not match"<<endl;
	    iz0=z0/DZ;
	  }


	  //cout << "Found good tracklet"<<endl;
	  //cout << " r z1 deltax deltaz: "<<r1<<" "<<z1<<" "
	  //     << fabs(x1-x2)<<" "<<fabs(z1-z2)<<endl;
	  //cout << "rinv rinv_n :"<<rinv<<" "<<rinv_n<<" "<<rinv_b<<endl;
	  //cout << "phi0 phi0_n+phic phi0_b+phic :"<<phi0<<" "<<phi0_n+phic<<" "<<phi0_b+phic<<endl;
	  //cout << "t t_n t_b      :"<<t<<" "<<t_n<<" "<<t_b<<endl;
	  //cout << "z0 z0_n z0_b    :"<<z0<<" "<<z0_n<<" "<<z0_b<<endl;

	  L1TTracklet trklet(i,j,rinv,phi0,z0,t,
			     irinv, iphi0, iz0, it,
			     layer_,local_rod_,phiSectorCenter_); 

	  trklet.addStubComponent(stubs_[i]);
	  trklet.addStubComponent(aRod.stubs_[j]);

	  tmp.addTracklet(trklet);

	}

	// cout << endl;


      }
    }
	
    return tmp;

  }


  void exactTracklet(double &rinv, double &phi0, double &z0, double &t,
		     double r1, double phi1, double z1, 
		     double r2, double phi2, double z2){
    
    static double two_pi=8*atan(1.0);
      
    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;

    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
        
    rinv=2*sin(deltaphi)/dist;

    phi0=phi1+asin(0.5*r1*rinv);

    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;

    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;

    t=(z1-z2)/(rhopsi1-rhopsi2);

    z0=z1-t*rhopsi1;

  }

  
  void approxTracklet(double &rinv_n, double &phi0_n, double &z0_n, double &t_n,
		      double r, double x1, double x2, double z1, double z2, 
		      double Delta){


    double delta=x2-x1;
    double deltaz=z2-z1;
    double s1=1.0/sqrt(delta*delta+Delta*Delta);
    double t1=x1*Delta/r;
    double t2=delta-t1;
    double t3=2*t2/(r+Delta);
    //double C=1.0/sqrt((1.0+(x1+delta)*(x1+delta)/((r+Delta)*(r+Delta)))*(1.0+x1*x1/(r*r)));
    //double C=1.0;
    double C=1-x1*x1/(r*r);
    rinv_n=s1*t3*C;
    double alpha=asin((x1/r)/sqrt(1+x1*x1/(r*r)));
    double beta=asin((0.5*r*rinv_n)*sqrt(1+x1*x1/(r*r)));
    phi0_n=beta-alpha;
    t_n=deltaz*s1;
    double rhit=sqrt(r*r+x1*x1);
    double t4=rinv_n*rinv_n;
    double t5=t4*rhit*rhit/24;
    double t6=1+t5;
    double t7=t_n*rhit;
    double t8=t7*t6;
    z0_n=z1-t8;

  }

  double round(double r) {
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
  }

  int lookup_is1(int idelta, int iDelta){

    int SHIFT=64;

    if (idelta<0) idelta=-idelta;

    idelta=(idelta/SHIFT)*SHIFT;

    static int max=0;

    if (idelta/SHIFT>max) {
      max=idelta/SHIFT;
      cout << "lookup_is1 max:"<<max<<endl;
    }

    return BASE2/sqrt(idelta*idelta+iDelta*iDelta);
     
  }

  int lookup_alpha(int ix1, double r){

    int SHIFT=4;

    bool negative=false;

    if (ix1<0) {
      ix1=-ix1;
      negative=true;
    }

    ix1=(ix1/SHIFT)*SHIFT;

    static int max=0;

    if (ix1/SHIFT>max) {
      max=ix1/SHIFT;
      cout << "lookup_alpha max:"<<max<<endl;
    }
    
    int alpha=BASE*asin((DX*ix1/r)/sqrt(1+DX*DX*ix1*ix1/(r*r))); 
    

    if (negative) {
      return -alpha;
    }

    return alpha;

  }

  int lookup_beta(int irinv, int ix1, double r){

    int SHIFT=128;

    bool negative=false;

    if (irinv<0) {
      irinv=-irinv;
      negative=true;
    }

    irinv=(irinv/SHIFT)*SHIFT;

    static int max=0;

    if (irinv/SHIFT>max) {
      max=irinv/SHIFT;
      cout << "lookup_beta max:"<<max<<endl;
    }


    int beta=BASE*asin((0.5*DX*r*irinv/BASE3)*sqrt(1+DX*DX*ix1*ix1/(r*r))); 

    //cout << "lookup_beta:"<<BASE<<" "<<DX<<" "<<r<<" "<<irinv<<" "<<BASE3
    //	 << " "<<ix1<<endl;
    
    if (negative) {
      return -beta;
    }

    return beta;

  }


  void binaryTracklet(double &rinv_n, double &phi0_n, double &z0_n, double &t_n,
		      int &irinv, int &iphi0, int &iz0, int &it,
		      double r, double x1, double x2, double z1, double z2, 
		      double Delta, double phic){


    //inputs to algorithm
    int iz1=round(z1/DZ);            //counted in 1.25 mm pixels.
    int idelta=round((x2-x1)/DX);   //counted in 50 um steps
    int ideltaz=round((z2-z1)/DZ);   //counted in 1.25 mm pixels
    int ix1=round(x1/DX);           //counted in 50 um steps

    //quantities that are precomputed for a given module
    int DeltaOverr=(Delta/r)*BASE5;
    int iDelta=round(Delta/DX);     //counted in 50 um steps
    int TwoOverOuterr=(2.0/((r+Delta)*DX))*BASE6;
    int DXOverr=(DX/r)*BASE7;
    int DXrsq=(DX*DX*r*r/24.0)*BASE4;
    int roverDX=r/DX;

    int is1=lookup_is1(idelta,iDelta); 
    int ia=ix1*DXOverr;
    int ib=ia*ia/BASE7;
    int ic=BASE7-ib; 
    int is2=is1*ic/BASE7;
    int it1=ix1*DeltaOverr/BASE5; 
    int it2=idelta-it1;         
    int it3=it2*TwoOverOuterr/BASE6; 
    irinv=is2*it3/(BASE2/BASE3);     
    int ialpha=lookup_alpha(ix1,r);
    int ibeta=lookup_beta(irinv,ix1,r);
    iphi0=ibeta-ialpha;      
    it=ideltaz*is1/(BASE2/BASE3);     
    int it4=irinv*irinv/(BASE3*BASE3/BASE2); 
    int it5=it4*DXrsq/(BASE4*(BASE2/BASE8));    
    int it6p=BASE8+ib/(2*BASE7/BASE8);
    int it6=it6p+it5;    
    int it7=it*roverDX/BASE3;     
    int it8=it6*it7/BASE8; 
    //cout << "it6 it7:"<<it6<<" "<<it7<<endl;
    iz0=iz1-it8;            



    rinv_n=(DX*irinv)/BASE3;
    phi0_n=iphi0/(1.0*BASE);
    t_n=(DZ/DX)*it/(1.0*BASE3);
    z0_n=iz0*DZ;

    iphi0+=(phic-phiSectorCenter_)*BASE;
    
    //cout << "Done calculating new variables"<<endl;
    
  }



  unsigned int matchTracklets(L1TTracklets& tracklets){

    //cout << "In L1TRod::matchTracklets N. of tracklets="
    //	 <<tracklets.size()<<" N. of stubs="<<stubs_.size()<<endl;

    unsigned int N_matches = 0;

    for (unsigned int i=0;i<tracklets.size();i++) {

      L1TTracklet& aTracklet=tracklets.get(i);

      int irinv=aTracklet.irinv();
      int iphi0=aTracklet.iphi0();
      int iz0=aTracklet.iz0();
      int it=aTracklet.it();

      double rinv=aTracklet.rinv();
      double phi0=aTracklet.phi0();
      double z0=aTracklet.z0();
      double t=aTracklet.t();

      //cout << "rinv="<<rinv<<" phi0="<<phi0<<endl;

      bool onRod=hitRod(rinv,phi0);

      if (!onRod) continue;
      
      double phic;
      double r1=getrmin(phic);

 
      int DXrsq=(DX*DX*r1*r1/24.0)*BASE4;
      int roverDX=r1/DX;

      //dropping x1 dependence. 
      int beta=lookup_beta(irinv,0,r1);
      int iphiproj=iphi0-beta;

      //cout << "phic phiSectorCenter_+iphiproj/(1.0*BASE) :"
      //	   << phic << " " << phiSectorCenter_+iphiproj/(1.0*BASE) << endl;

      double deltaphi=phic-(phiSectorCenter_+iphiproj/(1.0*BASE));

      if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
      if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;

      double x1=deltaphi*r1;
      if (fabs(x1)>6.0) {
	cout << "x1:"<<x1<<endl;
	cout << "r1:"<<r1<<endl;
	cout << "deltaphi:"<<deltaphi<<endl;
	cout << "phic:"<<phic<<endl;
	cout << "phicSectorCenter_:"<<phiSectorCenter_<<endl;
	cout << "iphiproj:"<<iphiproj<<endl;
	cout << "iphi0:"<<iphi0<<endl;
	cout << "beta:"<<beta<<endl;
	cout << "irinv:"<<irinv<<endl;
      }

      int ix1=round(x1/DX);
      int DXOverr=(DX/r1)*BASE9;

      //now calculate with knowing ix1
      iphiproj=iphi0-lookup_beta(irinv,ix1,r1);
      int ia=ix1*DXOverr;
      int ib=ia*ia/BASE9;
      int it4=irinv*irinv/(BASE3*BASE3/BASE2);
      int it5=it4*DXrsq/(BASE4*(BASE2/BASE8)); 
      int it6p=BASE8+ib/(2*BASE9/BASE8);
      int it6=it6p+it5;    
      int it7=it*roverDX/BASE3;     
      int it8=it6*it7/BASE8;    
      int izproj=iz0+it8;


      static ofstream outproj("proj.txt");

      if (onRod) {
	outproj //<< SLHCEvent::event << " " 
		<< sector_ << " " 
		<< rinv << " " 
		<<aTracklet.layer() <<" " 
		<<aTracklet.z() <<" " 
		<<aTracklet.local_rod()<< " "
		<<layer_ <<" "
		<<local_rod_<<" "
		<<izproj*DZ<< endl;
      }

      if (stubs_.size()==0) continue;
      
      L1TStub bestStub=stubs_[0];
      double bestDist=1e30;
      


      for (unsigned int j=0;j<stubs_.size();j++) {

	double r=stubs_[j].r();
	double z=stubs_[j].z();
	double phi=stubs_[j].phi();

	//double x=getx(r,phi);
	int iphihit=(phi-phiSectorCenter_)*BASE;

	int deltairphi=(iphiproj-iphihit)*(roverDX/BASE18);
	double tmp=two_pi*(roverDX/BASE18)*BASE;
	if (deltairphi>0.5*tmp) deltairphi-=tmp;
	if (deltairphi<-0.5*tmp) deltairphi+=tmp;

	//cout << "phi phiSectorCenter_:"<<phi<<" "<<phiSectorCenter_<<" "<<iphihit<<" "<<iphiproj<<" "<<iphi0<<" "<<deltairphi<<endl;

	//cout << "x1 t it8-2*t*asin(0.5*r*rinv)/rinv:"<<x1<<" "<<t<<" "<<it8*DZ-2*t*asin(0.5*r*rinv)/rinv<<endl;

	double phiproj=phi0-asin(0.5*r*rinv);
	double zproj=z0+2*t*asin(0.5*r*rinv)/rinv;

        if (1) {

	  static ofstream out("matchbinary.txt");
	  
	  out << aTracklet.r()
	      << " " << r
	      << " " << phi
	      << " " << phiproj
	      << " " << z
	      << " " << zproj
	      << " " << iphiproj
	      << " " << izproj*DZ
	      << " " << phiSectorCenter_+iphiproj/(1.0*BASE)
	      << " " << deltairphi*DX*BASE18/(BASE*r)
	      << " " << x1
	      << endl;

	}

        if (1) {

	  static ofstream out("match.txt");
	  out << aTracklet.r() <<" "<<rinv << " "<<r<<" "<<phi<<" "<<phiproj
	     << " "<<z<<" "<<zproj<<endl;

	}


	double rphicut=-1.0;
	double zcut=-1.0;

	if (aTracklet.r()<40.0){
	  if (r<60){
	    rphicut=0.075;
	    zcut=0.6;
	  }
	  else {
	    rphicut=0.5;
	    zcut=2.0;
	  }
	}
	else if (aTracklet.r()>60.0) {
	  if (r<40){
	    rphicut=0.15;
	    zcut=2.0;
	  }
	  else {
	    rphicut=0.15;
	    zcut=1.5;
	  }

	}
	else {
	  if (r<50){
	    rphicut=0.075;
	    zcut=0.6;
	  }
	  else {
	    rphicut=0.3;
	    zcut=2.0;
	  }

	}

	double deltaphi=phi-phiproj;
	
	if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;

	if (fabs(deltaphi)*r<rphicut&&fabs(z-zproj)<zcut) {
	  if (!onRod) cout << "Found match but was not on Rod:"
			   << sector_ <<" "<<layer_ <<" "<<local_rod_<<" "<<phiproj <<" "<< phi1_ <<" "<< phi2_<<endl;

	  double dist=hypot(fabs(phi-phiproj)*r/rphicut,
			    fabs(z-zproj)/zcut);

	  if (dist<bestDist){
	    //cout << "New best:"<<r<<" "<<deltaphi<<" "<<z-zproj
	    //	 << " "<<deltairphi<<" "<<izproj-z/DZ<<endl;

	    bestStub=stubs_[j];
	    bestStub.setideltarphi(deltairphi);
	    bestStub.setideltazi(izproj-z/DZ);
	    bestDist=dist;
	  }

	  //stubs_[j].setideltarphi(deltairphi);
	  //stubs_[j].setideltazi(izproj-z/DZ);
          //aTracklet.addStub(stubs_[j]); //not enough info
	  N_matches++;
	}

      }

      if (bestDist<1e29) {
	aTracklet.addStub(bestStub); //not enough info
      }

    }
    return N_matches;

  }

  unsigned int findCombinations(L1TTracklets& tracklets) {

    unsigned int N_combs = 0;

    for (unsigned int i=0;i<tracklets.size();i++) {
      N_combs += stubs_.size();
    }

    return N_combs; 
  }

  int nstubs() const { return stubs_.size(); }

  void printModuleMultiplicity() {

    map<int,int>::const_iterator it=stubmultiplicity_.begin();
    while (it!=stubmultiplicity_.end()) {  
      if (1) {
	static ofstream out("stubmultiplicity.txt");
	out << layer_ << " " << it->second << endl;
      }
      it++;
    }

    it=stubmultiplicityroc_.begin();
    while (it!=stubmultiplicityroc_.end()) {  
      if (1) {
	static ofstream out("stubmultiplicityroc.txt");
	out << layer_ << " " << it->second << endl;
      }
      it++;
    }

  }

  void addGeom(double r1,double phi1,double r2, double phi2, double phiSC){

    static double two_pi=8*atan(1.0);
    
    phiSectorCenter_=phiSC;

    r1_=r1;
    r2_=r2;
    phi1_=phi1;
    phi2_=phi2;
    if (phi1_<0) {
      phi1_+=two_pi;
    }
    if (phi2_<0) {
      phi2_+=two_pi;
    }
    if (fabs(phi1_-phi2_)>0.5*two_pi){
      if (phi1_<phi2_) phi1_+=two_pi;
      if (phi1_>phi2_) phi2_+=two_pi;
    }
    if (phi1_>phi2_) {
      double tmp=phi1_;
      phi1_=phi2_;
      phi2_=tmp;
      tmp=r1_;
      r1_=r2_;
      r2_=tmp;
    }
    
  }

  bool hitRod(double rinv,double phi0){
    static double two_pi=8*atan(1.0);
    double phiproj1=phi0-asin(0.5*r1_*rinv);
    double phiproj2=phi0-asin(0.5*r2_*rinv);
    if (phiproj1<0) {
      phiproj1+=two_pi;
    }
    if (phiproj2<0) {
      phiproj2+=two_pi;
    }

    bool hitTheRod=(phiproj1>phi1_-0.005&&phiproj2<phi2_+0.005);
    
    //if (hitTheRod) {
      //cout << "hitRod:";
    //}
    //else {
      //cout << "not hitRod:";
    //}
	 
    //cout <<"hitRod :"<<rinv<<" "<<phi0<<" "<<phi1_<<" "<<phi2_<<"   "<<phiproj1<<" "<<phiproj2<<endl;

    return hitTheRod;
    
  }


  double getx(double r,double phi) const {
    static double two_pi=8*atan(1.0);

    double x=r*cos(phi);
    double y=r*sin(phi);

    double x1=(r1_+0.06)*cos(phi1_);  //HACK for radius
    double y1=(r1_+0.06)*sin(phi1_);

    double x2=(r2_+0.06)*cos(phi2_);
    double y2=(r2_+0.06)*sin(phi2_);

    double l=sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
    double l1=sqrt((x1-x)*(x1-x)+(y1-y)*(y1-y));
    double l2=sqrt((x-x2)*(x-x2)+(y-y2)*(y-y2));

    double epsilon=(l1+l2)/l-1.0;

    if (fabs(epsilon)>0.01) {
      static int count=0;
      count++;
      if (count<100) {
	cout << "x,x1,x2,y,y1,y2:"<<x<<" "<<x1<<" "<<x2<<" "
	     <<y<<" "<<y1<<" "<<y2<<endl;
	cout << "l,l1,l2:"<<l<<" "<<l1<<" "<<l2<<endl;
      }
    }
    
    double phic=atan2(x2-x1,y1-y2);

    //cout << "phic :"<<phic<<endl;

    double deltaphi=phi-phic;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    
    double xtmp=r*sin(deltaphi);

    return xtmp;

  }

  double getrmin(double &phic) const {
    static double two_pi=8*atan(1.0);


    double x1=r1_*cos(phi1_);
    double y1=r1_*sin(phi1_);

    double x2=r2_*cos(phi2_);
    double y2=r2_*sin(phi2_);
    
    phic=atan2(x1-x2,y2-y1);

    double alpha=phic-phi1_;

    //cout << "phic phi1_ phi2_:"<<phic<<" "<<phi1_<<" "<<phi2_<<endl;

    if (alpha>0.5*two_pi) alpha-=two_pi;
    if (alpha<-0.5*two_pi) alpha+=two_pi;
    
    return cos(alpha)*(r1_+0.06);
    

  }


  double x1() const { return r1_*cos(phi1_); }
  double y1() const { return r1_*sin(phi1_); }

  double phiSectorCenter() { return phiSectorCenter_; }

private:

  int sector_;
  int layer_;
  int local_rod_;

  double r1_;
  double r2_;
  double phi1_;
  double phi2_;
  double phiSectorCenter_;

  vector<L1TStub> stubs_;
  map<int,int> stubmultiplicity_;
  map<int,int> stubmultiplicityroc_;

};



#endif



