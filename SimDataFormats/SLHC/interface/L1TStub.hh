#ifndef L1TSTUB_H
#define L1TSTUB_H

#include <iostream>
#include <assert.h>
using namespace std;

//This is the number of strips in rphi and in z for a module.
//This should be in the header of the ASCII file, but for now
//just hardcoded here.



class L1TStub{

public:

  L1TStub(){
  }

  L1TStub(int simtrackid, int iphi, int iz, int layer, int ladder, int module, 
	  double x, double y, double z, double sigmax, double sigmaz, double pt){
    simtrackid_=simtrackid;
    iphi_=iphi;
    iz_=iz;
    layer_=layer;
    ladder_=ladder;
    module_=module;
    x_=x;
    y_=y;
    z_=z;
    ideltarphi_=0;
    ideltaz_=0;
    sigmax_=sigmax;
    sigmaz_=sigmaz;
    pt_=pt;
    assert(z_<300.0); assert(z_>-300.0);
  }

  void lorentzcor(double shift){
    double r=this->r();
    double phi=this->phi()-shift/r;
    this->x_=r*cos(phi);
    this->y_=r*sin(phi);
  }

  

  double r() const { return sqrt(x_*x_+y_*y_); }
  double z() const { 
    assert(z_<300.0); assert(z_>-300.0); 
    return z_; 
  }
  double phi() const { return atan2(y_,x_); }
  unsigned int iphi() const { return iphi_; }
  unsigned int iz() const { return iz_; }

  unsigned int layer() const { return layer_;}
  unsigned int ladder() const { return ladder_;}
  unsigned int module() const { return module_;}

  int simtrackid() const { return simtrackid_;}

  bool operator== (const L1TStub& other) const {
    if (other.iphi()==iphi_ &&
	other.iz()==iz_ &&
	other.layer()==layer_ &&
	other.ladder()==ladder_ &&
	other.module()==module_)
      return true;

    else
      return false;
  }

  int ideltarphi() { return ideltarphi_;}
  int ideltaz() { return ideltaz_;}

  void setideltarphi(int ideltarphi) {ideltarphi_=ideltarphi;}
  void setideltazi(int ideltaz) {ideltaz_=ideltaz;}

  double sigmax() const {return sigmax_;}
  double sigmaz() const {return sigmaz_;}

  double pt() const {return pt_;}

private:

  int simtrackid_;
  unsigned int iphi_;
  unsigned int iz_;
  unsigned int layer_;
  unsigned int ladder_;
  unsigned int module_;
  double x_;
  double y_;
  double z_;
  double sigmax_;
  double sigmaz_;
  double pt_;

  int ideltarphi_;
  int ideltaz_;

};



#endif



