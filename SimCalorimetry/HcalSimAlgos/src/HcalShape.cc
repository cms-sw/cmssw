#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include <cmath>
  
HcalShape::HcalShape()
: nbin_(256),
  nt_(nbin_, 0.)
{
   setTpeak(32.0);
   computeShape();
}

HcalShape::HcalShape(const HcalShape&d):
  CaloVShape(d),
  nbin_(d.nbin_),
  nt_(d.nt_)
{
  setTpeak(32.0);
}


void HcalShape::computeShape()
{

  // pulse shape time constant_s in ns
  const float ts1  = 8.;          // scintillation time constant_s : 1,2,3
  const float ts2  = 10.;           
  const float ts3  = 29.3;         
  const float thpd = 4.;          // HPD current_ collection drift time
  const float tpre = 9.;          // preamp time constant_ (refit on TB04 data)
  
  const float wd1 = 2.;           // relative weights of decay exponent_s 
  const float wd2 = 0.7;
  const float wd3 = 1.;
  
  // pulse shape componnt_s over a range of time 0 ns to 255 ns in 1 ns steps
  std::vector<float> nth(nbin_,0.0);   // zeroing HPD drift shape
  std::vector<float> ntp(nbin_,0.0);   // zeroing Binkley preamp shape
  std::vector<float> ntd(nbin_,0.0);   // zeroing Scintillator decay shape

  int i,j,k;
  float norm;

  // HPD starts at I and rises to 2I in thpd of time
  norm=0.0;
  for(j=0;j<thpd && j<nbin_;j++){
    nth[j] = 1.0 + j/thpd;
    norm += nth[j];
  }
  // normalize integrated current_ to 1.0
  for(j=0;j<thpd && j<nbin_;j++){
    nth[j] /= norm;
  }
  
  // Binkley shape over 6 time constant_s
  norm=0.0;
  for(j=0;j<6*tpre && j<nbin_;j++){
    ntp[j] = j*exp(-(j*j)/(tpre*tpre));
    norm += ntp[j];
  }
  // normalize pulse area to 1.0
  for(j=0;j<6*tpre && j<nbin_;j++){
    ntp[j] /= norm;
  }

// ignore stochastic variation of photoelectron emission
// <...>

// effective tile plus wave-length shifter decay time over 4 time constant_s
  int tmax = 6 * static_cast<int>(ts3);
 
  norm=0.0;
  for(j=0;j<tmax && j<nbin_;j++){
    ntd[j] = wd1 * exp(-j/ts1) + 
      wd2 * exp(-j/ts2) + 
      wd3 * exp(-j/ts3) ; 
    norm += ntd[j];
  }
  // normalize pulse area to 1.0
  for(j=0;j<tmax && j<nbin_;j++){
    ntd[j] /= norm;
  }
  
  int t1,t2,t3,t4;
  for(i=0;i<tmax && i<nbin_;i++){
    t1 = i;
    //    t2 = t1 + top*rand;
    // ignoring jitter from optical path length
    t2 = t1;
    for(j=0;j<thpd && j<nbin_;j++){
      t3 = t2 + j;
      for(k=0;k<4*tpre && k<nbin_;k++){       // here "4" is set deliberately,
 t4 = t3 + k;                         // as in test fortran toy MC ...
 if(t4<nbin_){                         
   int ntb=t4;                        
   nt_[ntb] += ntd[i]*nth[j]*ntp[k];
	}
      }
    }
  }
  
  // normalize for 1 GeV pulse height
  norm = 0.;
  for(i=0;i<nbin_;i++){
    norm += nt_[i];
  }

  //cout << " Convoluted SHAPE ==============  " << endl;
  for(i=0; i<nbin_; i++){
    nt_[i] /= norm;
    //  cout << " shape " << i << " = " << nt_[i] << endl;   
  }

}

double HcalShape::operator () (double time_) const
{

  // return pulse amplitude for request time in ns
  int jtime = static_cast<int>(time_+0.5);
  if(jtime>=0 && jtime<nbin_){
    return nt_[jtime];
  } else {
    return 0.0;
  }

}

double HcalShape::derivative (double time_) const
{
  return 0.0;
}


