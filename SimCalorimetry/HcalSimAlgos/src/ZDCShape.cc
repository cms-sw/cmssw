#include "SimCalorimetry/HcalSimAlgos/interface/ZDCShape.h"
#include <cmath>
  
ZDCShape::ZDCShape()
: nbin_(256),
  nt_(nbin_, 0.)
{   
  computeShapeZDC();
}


ZDCShape::ZDCShape(const ZDCShape&d)
: CaloVShape(d),
  nbin_(d.nbin_),
  nt_(d.nt_)
{
}

double
ZDCShape::timeToRise() const 
{
   return 0. ;
}

  
void ZDCShape::computeShapeZDC()
{

  //  cout << endl << " ===== computeShapeZDC  !!! " << endl << endl;

  const float ts = 3.0;           // time constant in   t * exp(-(t/ts)**2)


  int j;
  float norm;

  // ZDC SHAPE
  norm = 0.0;
  for( j = 0; j < 3 * ts && j < nbin_; j++){
    //nt_[j] = ((float)j)*exp(-((float)(j*j))/(ts*ts));
    nt_[j] = j * exp(-(j*j)/(ts*ts));
    norm += nt_[j];
  }
  // normalize pulse area to 1.0
  for( j = 0; j < 3 * ts && j < nbin_; j++){
    nt_[j] /= norm;
  }
}

double ZDCShape::operator () (double time) const
{

  // return pulse amplitude for request time in ns
  int jtime;
  jtime = static_cast<int>(time+0.5);

  if(jtime >= 0 && jtime < nbin_)
    return nt_[jtime];
  else 
    return 0.0;
}
  
