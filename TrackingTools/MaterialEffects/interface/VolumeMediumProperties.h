#ifndef VOLUMEMEDIUMPROPERTIES_H_
#define VOLUMEMEDIUMPROPERTIES_H_

/** \class VolumeMediumProperties
 *  Holds constants for estimating material effects in a volume:
 *   x0 = rad. length (in cm; used for estimation of multiple scattering)
 *   xi = rho(g/cm3) *0.307075[MeV/(g/cm2)] * <Z/A> * 1/2
 *                 (used for energy loss acc. to Bethe-Bloch)
 */

class VolumeMediumProperties
{  
public:
  VolumeMediumProperties (float x0, float xi) :
    x0_(x0), xi_(xi) {}

  // Radiation length (in cm)
  float x0 () const {return x0_;}
  // Scaling factor for energy loss (see class description)
  float xi () const {return xi_;}

 private:
  float x0_;
  float xi_;
};

#endif
