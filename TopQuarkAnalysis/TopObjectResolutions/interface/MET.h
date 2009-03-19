#ifndef TopObjetcResolutionsMET_h
#define TopObjetcResolutionsMET_h

namespace res{
  class HelperMET {
    
  public:
    HelperMET(){};
    ~HelperMET(){};
    
    inline double met(double met);
    inline double a(double pt);
    inline double b(double pt);
    inline double c(double pt);
    inline double d(double pt);
    inline double theta(double pt);
    inline double phi(double pt);
    inline double et(double pt);
    inline double eta(double pt);
  };
}

inline double res::HelperMET::met(double met)
{
  return 1.14*exp(-2.16e-3*met)+0.258;
}

inline double res::HelperMET::a(double pt)
{
	double res = 0.241096+0.790046*exp(-(0.0248773*pt));
  return res;
}

inline double res::HelperMET::b(double pt)
{
	double res = -141945+141974*exp(-(-1.20077e-06*pt));
  return res;
}

inline double res::HelperMET::c(double pt)
{
	double res = 21.5615+1.13958*exp(-(-0.00921408*pt));
  return res;
}

inline double res::HelperMET::d(double pt)
{
	double res = 0.376192+15.2485*exp(-(0.116907*pt));
  return res;
}

inline double res::HelperMET::theta(double pt)
{
	double res = 1000000.;
  return res;
}

inline double res::HelperMET::phi(double pt)
{
	double res = 0.201336+1.53501*exp(-(0.0216707*pt));
  return res;
}

inline double res::HelperMET::et(double pt)
{
	double res = 11.7801+0.145218*pt;
  return res;
}

inline double res::HelperMET::eta(double pt)
{
	double res = 1000000.;
  return res;
}

#endif
