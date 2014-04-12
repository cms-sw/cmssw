#ifndef global_h
#define global_h
#include <TROOT.h>
#include <math.h>
//
// define some useful constants:
//
const Double_t sqrt2    = sqrt(2.0);
const Double_t sqrt2pi  = 2.5066282746;

const int dim    = 11;  // number of fit parameters
// 'pointers' to the parameters
const int Par_Z0    = 0;  // index of position of luminosity peak in z  
const int Par_Sigma = 1;  // index of sigma in z of the beam

const int Par_x0    = 2;  // beam spot x
const int Par_y0    = 3;  // beam spot y
const int Par_dxdz  = 4;  // beam slop dxdz
const int Par_dydz  = 5;  // beam slop dydz
const int Par_Sigbeam=6;  // beam spot width

const int Par_c0    = 7;
const int Par_c1    = 8;

const int Par_eps   = 9;  // index of emittance 
const int Par_beta  = 10;  // index of beta*

struct data
 {
   double Z,SigZ,D,SigD,Phi,Pt,weight2;
   data(double z, double sigz,double d, double sigd,double phi,
		double pt,double weight2)
     : Z(z), SigZ(sigz),D(d),SigD(sigd),Phi(phi),Pt(pt),weight2(1){}
};
typedef std::vector<data> zData;//!
typedef zData::const_iterator zDataConstIter;//!
typedef zData::iterator zDataIter;//!
//zData zdata;//!
#endif //
