#ifndef TopObjetcResolutionsMuon_h
#define TopObjetcResolutionsMuon_h

namespace res{
  class HelperMuon {

  public:
    HelperMuon(){};
    ~HelperMuon(){};

    inline double pt (double pt, double eta);
    inline double eta(double pt, double eta);
    inline double phi(double pt, double eta);
  };
}

inline double res::HelperMuon::pt(double pt, double eta)
{
  return 1.5e-4*(pt*pt)+0.534*fabs(eta)+1.9e-2;
}

inline double res::HelperMuon::eta(double pt, double eta)
{
  return 6.2e-5*eta*eta-2e-4*fabs(eta)+4e-4; 
}

inline double res::HelperMuon::phi(double pt, double eta)
{
  return 3.7e-5*fabs(eta)+1.4e-4;
}

#endif
