#ifndef TopObjetcResolutionsElectron_h
#define TopObjetcResolutionsElectron_h

namespace res{
  class HelperElectron {

  public:
    HelperElectron(){};
    ~HelperElectron(){};

    inline double pt (double pt, double eta);
    inline double eta(double pt, double eta);
    inline double phi(double pt, double eta);
  };
}

inline double res::HelperElectron::pt(double pt, double eta)
{
  double res=0.2*sqrt(pt);
  if( fabs(eta)<=0.8 ) 
    res+=-0.28*fabs(eta)+0.54;
  else if( 0.8<fabs(eta) && fabs(eta)<=1.4 ) 
    res+= 1.52*fabs(eta)-1.07;
  else  
    res+=-0.158*eta*eta +0.97;
  return res;
}

inline double res::HelperElectron::eta(double pt, double eta)
{
  return -8.5e-5*fabs(eta)+4e-4;
}

inline double res::HelperElectron::phi(double pt, double eta)
{
  return 7.6e-5*eta*eta-7.7e-5*fabs(eta)+5.8e-4;
}

#endif
