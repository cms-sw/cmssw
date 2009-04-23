#ifndef TopObjetcResolutionsMET_h
#define TopObjetcResolutionsMET_h

namespace res{
  class HelperMET {
    
  public:
    HelperMET(){};
    ~HelperMET(){};
    
    inline double met(double met);
    inline double phi(double met);
  };
}

inline double res::HelperMET::met(double met)
{
  return 1.14*exp(-2.16e-3*met)+0.258;
}

inline double res::HelperMET::phi(double met)
{
  return 1.35e-3*met*met+0.137*met+1.454;
}

#endif
