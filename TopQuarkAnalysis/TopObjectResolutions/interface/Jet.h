#ifndef TopObjetcResolutionsJet_h
#define TopObjetcResolutionsJet_h

namespace res{
  class HelperJet {

  public:
    enum Flavor {kUds, kB};

    HelperJet(){};
    ~HelperJet(){};

    inline double pt (double pt, double eta,  Flavor flav);
    inline double eta(double pt, double eta,  Flavor flav);
    inline double phi(double pt, double eta,  Flavor flav);
  };
}

inline double res::HelperJet::pt(double pt, double eta,  Flavor flav)
{
  double res = 0.29*sqrt(pt);
  if(fabs(eta)<1.4) res+= 6.68;
  else res+=-3.14*fabs(eta)+11.89;
  if(flav==kB){
    res=0.33*sqrt(pt);
    if(fabs(eta)<1.4) res+= 6.57;
    else res+=-1.09*fabs(eta)+8.50;    
  }
  return res;
}

inline double res::HelperJet::eta(double pt, double eta,  Flavor flav)
{
  double res=-1.53e-4*pt+0.05;
  if(flav==kB)
    res=-1.2e-4*pt+0.05;
  return res;
}

inline double res::HelperJet::phi(double pt, double eta,  Flavor flav)
{
  double res=-2.7e-4*pt+0.06;
  if(flav==kB)
    res=-2.1e-4*pt+0.05;
  return res;
}

#endif
