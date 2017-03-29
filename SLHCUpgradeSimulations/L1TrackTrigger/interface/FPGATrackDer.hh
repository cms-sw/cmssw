#ifndef FPGATRACKDER_H
#define FPGATRACKDER_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>


using namespace std;

class FPGATrackDer{

public:

  FPGATrackDer() {

  }

  ~FPGATrackDer() {

  }

  void setIndex(int layermask,int diskmask,int alphamask){

    layermask_=layermask;
    diskmask_=diskmask;
    alphamask_=alphamask;

  }

  int getLayerMask() const {return layermask_;}
  int getDiskMask() const {return diskmask_;}
  int getAlphaMask() const {return alphamask_;}

  void setirinvdphi(int i, int irinvdphi) { irinvdphi_[i]=irinvdphi;} 
  void setirinvdzordr(int i, int irinvdzordr) { irinvdzordr_[i]=irinvdzordr;} 
  void setiphi0dphi(int i, int iphi0dphi) { iphi0dphi_[i]=iphi0dphi;} 
  void setiphi0dzordr(int i, int iphi0dzordr) { iphi0dzordr_[i]=iphi0dzordr;} 
  void setitdphi(int i, int itdphi) { itdphi_[i]=itdphi;} 
  void setitdzordr(int i, int itdzordr) { itdzordr_[i]=itdzordr;} 
  void setiz0dphi(int i, int iz0dphi) { iz0dphi_[i]=iz0dphi;} 
  void setiz0dzordr(int i, int iz0dzordr) { iz0dzordr_[i]=iz0dzordr;} 

  void setrinvdphi(int i, double rinvdphi) { rinvdphi_[i]=rinvdphi;} 
  void setrinvdzordr(int i, double rinvdzordr) { rinvdzordr_[i]=rinvdzordr;} 
  void setphi0dphi(int i, double phi0dphi) { phi0dphi_[i]=phi0dphi;} 
  void setphi0dzordr(int i, double phi0dzordr) { phi0dzordr_[i]=phi0dzordr;} 
  void settdphi(int i, double tdphi) { tdphi_[i]=tdphi;} 
  void settdzordr(int i, double tdzordr) { tdzordr_[i]=tdzordr;} 
  void setz0dphi(int i, double z0dphi) { z0dphi_[i]=z0dphi;} 
  void setz0dzordr(int i, double z0dzordr) { z0dzordr_[i]=z0dzordr;} 

  double getrinvdphi(int i) const { return rinvdphi_[i];} 
  double getrinvdzordr(int i) const { return rinvdzordr_[i];} 
  double getphi0dphi(int i) const { return phi0dphi_[i];} 
  double getphi0dzordr(int i) const { return phi0dzordr_[i];} 
  double gettdphi(int i) const { return tdphi_[i];} 
  double gettdzordr(int i) const { return tdzordr_[i];} 
  double getz0dphi(int i) const { return z0dphi_[i];} 
  double getz0dzordr(int i) const { return z0dzordr_[i];} 

  void sett(double t) { t_=t; }
  double gett() const { return t_; }

  void fill(int t, double MinvDt[4][12], int iMinvDt[4][12]){
    unsigned int nlayer=0;
    if (layermask_&1) nlayer++;
    if (layermask_&2) nlayer++;
    if (layermask_&4) nlayer++;
    if (layermask_&8) nlayer++;
    if (layermask_&16) nlayer++;
    if (layermask_&32) nlayer++;
    int sign=1;
    if (t<0) sign=-1;
    for (unsigned int i=0;i<6;i++){
      if (i<nlayer) {
	MinvDt[0][2*i]=rinvdphi_[i];
	MinvDt[1][2*i]=phi0dphi_[i];
	MinvDt[2][2*i]=sign*tdphi_[i];
	MinvDt[3][2*i]=sign*z0dphi_[i];
	MinvDt[0][2*i+1]=sign*rinvdzordr_[i];
	MinvDt[1][2*i+1]=sign*phi0dzordr_[i];
	MinvDt[2][2*i+1]=tdzordr_[i];
	MinvDt[3][2*i+1]=z0dzordr_[i];
	iMinvDt[0][2*i]=irinvdphi_[i];
	iMinvDt[1][2*i]=iphi0dphi_[i];
	iMinvDt[2][2*i]=sign*itdphi_[i];
	iMinvDt[3][2*i]=sign*iz0dphi_[i];
	iMinvDt[0][2*i+1]=sign*irinvdzordr_[i];
	iMinvDt[1][2*i+1]=sign*iphi0dzordr_[i];
	iMinvDt[2][2*i+1]=itdzordr_[i];
	iMinvDt[3][2*i+1]=iz0dzordr_[i];
      } else {
	MinvDt[0][2*i]=rinvdphi_[i];
	MinvDt[1][2*i]=phi0dphi_[i];
	MinvDt[2][2*i]=sign*tdphi_[i];
	MinvDt[3][2*i]=sign*z0dphi_[i];
	MinvDt[0][2*i+1]=rinvdzordr_[i];
	MinvDt[1][2*i+1]=phi0dzordr_[i];
	MinvDt[2][2*i+1]=sign*tdzordr_[i];
	MinvDt[3][2*i+1]=sign*z0dzordr_[i];
	iMinvDt[0][2*i]=irinvdphi_[i];
	iMinvDt[1][2*i]=iphi0dphi_[i];
	iMinvDt[2][2*i]=sign*itdphi_[i];
	iMinvDt[3][2*i]=sign*iz0dphi_[i];
	iMinvDt[0][2*i+1]=irinvdzordr_[i];
	iMinvDt[1][2*i+1]=iphi0dzordr_[i];
	iMinvDt[2][2*i+1]=sign*itdzordr_[i];
	iMinvDt[3][2*i+1]=sign*iz0dzordr_[i];
      }
    }
  }


private:

  int irinvdphi_[6]; 
  int irinvdzordr_[6]; 
  int iphi0dphi_[6]; 
  int iphi0dzordr_[6]; 
  int itdphi_[6]; 
  int itdzordr_[6]; 
  int iz0dphi_[6]; 
  int iz0dzordr_[6]; 

  double rinvdphi_[6]; 
  double rinvdzordr_[6]; 
  double phi0dphi_[6]; 
  double phi0dzordr_[6]; 
  double tdphi_[6]; 
  double tdzordr_[6]; 
  double z0dphi_[6]; 
  double z0dzordr_[6]; 

  double t_;

  int layermask_;
  int diskmask_;
  int alphamask_;

};



#endif
