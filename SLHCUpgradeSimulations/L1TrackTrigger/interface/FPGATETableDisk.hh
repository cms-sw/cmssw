#ifndef FPGATETABLEDISK_H
#define FPGATETABLEDISK_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>


using namespace std;

class FPGATETableDisk{

public:

  FPGATETableDisk() {
   
  }

  ~FPGATETableDisk() {

  }


  void init(int stubpt1bits,
	    int stubpt2bits,
	    int deltaphibits,
	    int deltarbits,
	    int z1bits,
	    int z2bits,
	    int r1bits,
	    int r2bits,
	    double deltaphi,  //width of vm in phi
	    double deltaphioffset,  //offset of l2 vm with respect l1
            double zmin1,
	    double zmax1,
            double zmin2,
	    double zmax2,
            double rmin1,
	    double rmax1,
            double rmin2,
	    double rmax2,
	    int i1,
	    int i2,
	    int j1,
	    int j2
	    ) {

    //cout << "In FPGATETableDisk::init()"<<endl;
    
    z1bits_=z1bits;
    z2bits_=z2bits;
    r1bits_=r1bits;
    r2bits_=r2bits;
    
    i1_=i1;
    i2_=i2;
    j1_=j1;
    j2_=j2;


    phitablebits_=stubpt1bits+stubpt2bits+deltaphibits+deltarbits;
    ztablebits_=z1bits+z2bits+r1bits+r2bits;

    //int firstprint=true;

    phitableentries_=1<<phitablebits_;
    ztableentries_=1<<ztablebits_;

    //cout << "init 2 phitablebits_ : "<<phitablebits_<<endl;
    //cout << "init 2 phitableentries_ : "<<phitableentries_<<endl;


    for (int i=0;i<phitableentries_;i++){
      int istubpt1=i>>(stubpt2bits+deltaphibits+deltarbits);
      int istubpt2=(i>>(deltaphibits+deltarbits))&((1<<stubpt2bits)-1);
      int ideltaphi=(i>>(deltarbits))&((1<<deltaphibits)-1);
      int ideltar=i&((1<<deltarbits)-1);

      assert(istubpt1>=0&&istubpt1<8);
      assert(istubpt2>=0&&istubpt2<8);

      
      bool print=false;
      //if (i==10449) {
      //	print=true;
      //}
      

      if (ideltaphi>7) ideltaphi-=16;
      if (ideltar>7) ideltar-=16;

      double deltaphiavg=deltaphioffset+2*(deltaphi*ideltaphi)/(1<<deltaphibits);
      double deltar=rmin2-rmin1+2*ideltar*(rmax1-rmin1)/(1<<deltarbits);
      double Delta=sqrt(deltar*deltar+2*rmax1*rmax2*(1-cos(deltaphiavg)));
      double rinv=-2*sin(deltaphiavg)/Delta; //Minus sign is a hack

      if (print) cout << "deltar "<<deltar<<endl;

      double ptstubinv1;
      double ptstubinv2;
      if (istubpt1==0) ptstubinv1=0.40;
      if (istubpt1==1) ptstubinv1=0.25;
      if (istubpt1==2) ptstubinv1=0.15;
      if (istubpt1==3) ptstubinv1=0.05;
      if (istubpt1==4) ptstubinv1=-0.05;
      if (istubpt1==5) ptstubinv1=-0.15;
      if (istubpt1==6) ptstubinv1=-0.25;
      if (istubpt1==7) ptstubinv1=-0.40;

      if (istubpt2==0) ptstubinv2=0.40;
      if (istubpt2==1) ptstubinv2=0.25;
      if (istubpt2==2) ptstubinv2=0.15;
      if (istubpt2==3) ptstubinv2=0.05;
      if (istubpt2==4) ptstubinv2=-0.05;
      if (istubpt2==5) ptstubinv2=-0.15;
      if (istubpt2==6) ptstubinv2=-0.25;
      if (istubpt2==7) ptstubinv2=-0.40;

      double cut=teptconsistencydisk;

      if (rmin1<30.0) cut=cut*2.0;

      double pttracklet=0.3*3.8/(rinv*100);
      bool pass1=fabs(ptstubinv1-1.0/pttracklet)<cut;
      bool pass2=fabs(ptstubinv2-1.0/pttracklet)<cut;
      bool pass=pass1&&pass2;      



      double deltaphimin=2*(deltaphi*ideltaphi)/(1<<deltaphibits);
      deltaphimin+=deltaphioffset;
      deltaphimin=fabs(deltaphimin)-deltaphi/(1<<deltaphibits);
      if (deltaphimin<0.0) deltaphimin=0.0;

      //double deltarmax=rmin2-rmin1+0.5*((rmax1-rmin1)+(rmax2-rmin2))*
      //fabs(ideltar)/(1<<deltarbits);
      double deltarmax=rmin2-rmin1+2*ideltar*(rmax1-rmin1)/(1<<deltarbits);
     
      double Deltamax=sqrt(deltarmax*deltarmax+
			   2*rmax1*rmax2*(1-cos(deltaphimin)));
      double rinvmin=2*sin(deltaphimin)/Deltamax;

      //tablephi_.push_back((rinvmin<0.0057));
      tablephi_.push_back((rinvmin<0.0057)&&pass);


      
      if (print) cout << "deltaphimin deltaphi deltaphioffset ideltaphi rinvmin:"
		      << deltaphimin <<" "
		      << deltaphi<<" "
		      <<deltaphioffset <<" "
		      <<ideltaphi <<" "
		      <<rinvmin<<endl;

     }

    //cout << "init 3 phitablebits_ : "<<phitablebits_<<endl;


    for (int i=0;i<ztableentries_;i++){
      int iz1=i>>(z2bits+r1bits+r2bits);
      int iz2=i>>(r1bits+r2bits)&((1<<z2bits)-1);
      int ir1=i>>(r2bits)&((1<<r1bits)-1);
      int ir2=i&((1<<r2bits)-1);

      //bool printz=(iz1==3)&&(iz2==0)&&(ir1==29)&&(ir2==10);


      double z1[2];
      double z2[2];
      double r1[2];
      double r2[2];

      assert(zmin1<zmax1);
      assert(zmin2<zmax2);

      z1[0]=zmin1+iz1*(zmax1-zmin1)/(1<<z1bits);
      z1[1]=zmin1+(iz1+1)*(zmax1-zmin1)/(1<<z1bits);

      z2[0]=zmin2+iz2*(zmax2-zmin2)/(1<<z2bits);
      z2[1]=zmin2+(iz2+1)*(zmax2-zmin2)/(1<<z2bits);

      r1[0]=rmin1+ir1*(rmax1-rmin1)/(1<<r1bits);
      r1[1]=rmin1+(ir1+1)*(rmax1-rmin1)/(1<<r1bits);

      r2[0]=rmin2+ir2*(rmax2-rmin2)/(1<<r2bits);
      r2[1]=rmin2+(ir2+1)*(rmax2-rmin2)/(1<<r2bits);

      bool below=false;
      bool center=false;
      bool above=false;

      for(int iz1=0;iz1<2;iz1++){
	for(int iz2=0;iz2<2;iz2++){
	  for(int ir1=0;ir1<2;ir1++){
	    for(int ir2=0;ir2<2;ir2++){
	      double z0=z1[iz1]+(z1[iz1]-z2[iz2])*r1[ir1]/(r2[ir2]-r1[ir1]);
	      
	      //if (printz&&z1[0]<-185&&r1[0]<50.0) {
	      //cout << "z1 z2 r1 r2 z0 "
	      //     <<z1[iz1]<<" "
	      //     <<z2[iz2]<<" "
	      //     <<r1[ir1]<<" "
	      //     <<r2[ir2]<<" "
	      //     <<z0<<endl;
	      //}
	     
	      if (fabs(z0)<15.0) center=true;
	      if (z0<-15.0) below=true;
	      if (z0>15.0) above=true;
	    }
	  }
	}
      }

      //if (printz) {
      //	cout << "coords "
      //     <<0.5*(z1[0]+z1[1])<<" "
      //     <<0.5*(z2[0]+z2[1])<<" "
      //     <<0.5*(r1[0]+r1[1])<<" "
      //     <<0.5*(r2[0]+r2[1])<<" "<<(center||(below&&above))<<endl;
      //}

      tablez_.push_back(center||(below&&above));
      
      //cout << "i iz1 iz2 ir1 ir2 : "
      //   << i<<" "<<iz1<<" "<<iz2<<" "<<ir1<<" "<<ir2<<" "<<tablez_[i]<<endl;
    }

    //cout << "init 4 phitablebits_ : "<<phitablebits_<<endl;
    //cout << "init 2 phitableentries_ : "<<phitableentries_<<endl;


  }
	    

  void writephi(std::string fname) {

    ofstream out(fname.c_str());

    //cout << "writephi 2 phitableentries_ : "<<phitableentries_<<endl;

    for (int i=0;i<phitableentries_;i++){
      FPGAWord entry;
      //cout << "phitablebits_ : "<<phitablebits_<<endl;
      entry.set(i,phitablebits_);
      //out << entry.str()<<" "<<tablephi_[i]<<endl;
      out <<tablephi_[i]<<endl;
    }
    out.close();
      
  }


  void writer(std::string fname) {

    ofstream out(fname.c_str());

    for (int i=0;i<ztableentries_;i++){
      FPGAWord entry;
      //cout << "ztablebits_ : "<<ztablebits_<<endl;
      entry.set(i,ztablebits_);
      out << entry.str()<<" "<<tablez_[i]<<endl;
    }
      
    out.close();
  }

  bool phicheck(int address) const {
    assert(address>=0);
    assert(address<phitableentries_);
    return tablephi_[address];
  }

  bool zcheck(int z1, int z2, int r1, int r2) const {

    //cout << "z2 z2bits : "<<z2<<" "<<z2bits_<<endl; 
    //cout << "r1 r1bits : "<<r1<<" "<<r1bits_<<endl; 

    assert(z1>=0);
    assert(z2>=0);
    assert(r1>=0);
    assert(r2>=0);
    assert(z1<(1<<z1bits_));
    assert(z2<(1<<z2bits_));
    assert(r1<(1<<r1bits_));
    assert(r2<(1<<r2bits_));
	   
    int address=(z1<<(z2bits_+r1bits_+r2bits_))+
      (z2<<(r1bits_+r2bits_))+
      (r1<<r2bits_)+
      r2;

    assert(address>=0);
    assert(address<ztableentries_);
    return tablez_[address];
  }


private:

  vector<bool> tablephi_;
  vector<bool> tablez_;

  int phitableentries_;
  int ztableentries_;
  int phitablebits_;
  int ztablebits_;
  
  //int stubpt1bits_;
  //int stubpt2bits_;
  //int deltaphibits_;
  //int deltarbits_;
  int z1bits_;
  int z2bits_;
  int r1bits_;
  int r2bits_;
  
  int i1_;
  int i2_;
  int j1_;
  int j2_;

};



#endif



