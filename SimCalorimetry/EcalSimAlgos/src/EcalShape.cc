#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"
#include <cmath>


EcalShape::EcalShape()
{
  setTpeak(47.6683);

  // first create pulse shape over a range of time 0 ns to 255 ns in 1 ns steps
  // tconv give integer fraction of 1 ns
  tconv = 10;
  nbin = 256*tconv;
  std::vector<float> ntmp(nbin,0.0);  // zero output pulse shape
  std::vector<float> ntmpd(nbin,0.0);  // zero output deriveative pulse shape

  const double Par0=    1.0 ;   // In GeV Units
  const double Par1=  1.0/0.115855;
  // with a peaking time of 47.6683 ns
  const double Par2=   -1.3924 ;
  const double Par3=   0.34451 ;
  const double Par4=    182.64 ;
  const double Par5=   -2.1681 ;
  const double Par6=   0.67193 ;
  const double Par8=    170.59 ;

  int j;
  double xb,xf;
  double value,deriv;

  for(j=0;j<nbin;j++){
    xb = ((double)j)/tconv;
    value = 0.0;
    deriv = 0.0;
    if (xb>0.00001 && xb<Par8){
      xf = xb/Par4;
      value = Par1*( xf + Par2*xf*xf + Par3*xf*xf*xf )
      *exp( -xf + Par5*xf*xf + Par6*xf*xf*xf )/Par0;
      deriv = Par1*( xf + Par2*xf*xf + Par3*xf*xf*xf )
      *exp( -xf + Par5*xf*xf + Par6*xf*xf*xf )/Par0
      *( -1.0 + 2.0*Par5*xf + 3.0*Par6*xf*xf )/Par4
      +Par1*( 1.0 + 2.0*Par2*xf + 3.0*Par3*xf*xf )/Par4
      *exp( -xf + Par5*xf*xf + Par6*xf*xf*xf )/Par0;
    }
    ntmp[j] = (float)value;
    ntmpd[j] = (float)deriv;
  }

  /*
  for(i=0;i<nbin;i++){
    cout << i << " ECAL pulse shape " << ntmp[i] <<endl;
  }
  */

  nt = ntmp;
  ntd = ntmpd;

}

double EcalShape::operator () (double time_) const
{

  // return pulse amplitude for request time in ns
  int jtime;
  jtime = (int)(time_*tconv+0.5);
  if(jtime>=0 && jtime<nbin){
    return nt[jtime];
  } else {
    return 0.0;
  }

}


double EcalShape::derivative (double time_) const
{

  // return pulse amplitude for request time in ns
  int jtime;
  jtime = (int)(time_*tconv+0.5);
  if(jtime>=0 && jtime<nbin){
    return ntd[jtime];
  } else {
    return 0.0;
  }

}

