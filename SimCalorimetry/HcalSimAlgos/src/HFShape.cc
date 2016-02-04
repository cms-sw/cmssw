#include "SimCalorimetry/HcalSimAlgos/interface/HFShape.h"
#include <cmath>
  
HFShape::HFShape()
: nbin_(256),
  nt_(nbin_, 0.)
{   
  computeShapeHF();
}


HFShape::HFShape(const HFShape&d)
: CaloVShape(d),
  nbin_(d.nbin_),
  nt_(d.nt_)
{
}

double
HFShape::timeToRise() const 
{
   return 0. ;
}
  
void HFShape::computeShapeHF()
{

  //  cout << endl << " ===== computeShapeHF  !!! " << endl << endl;

  const float k0=0.7956; // shape parameters
  const float p2=1.355;
  const float p4=2.327;
  const float p1=4.3;    // position parameter

  int j;
  float norm,r0,sigma0;

  // HF SHAPE
  norm = 0.0;
  for( j = 0; j < 25 && j < nbin_; j++){

    r0 = j-p1;
    if (r0<0) sigma0 = p2;
    else sigma0 =  p2*p4;
    r0 = r0/sigma0;
    if(r0 < k0) nt_[j] = exp(-0.5*r0*r0);
    else nt_[j] = exp(0.5*k0*k0-k0*r0);
    norm += nt_[j];
  }
  // normalize pulse area to 1.0
  for( j = 0; j < 25 && j < nbin_; j++){
    nt_[j] /= norm;
  }
}

double HFShape::operator () (double time) const
{

  // return pulse amplitude for request time in ns
  int jtime;
  jtime = static_cast<int>(time+0.5);

  if(jtime >= 0 && jtime < nbin_)
    return nt_[jtime];
  else 
    return 0.0;
}
  
