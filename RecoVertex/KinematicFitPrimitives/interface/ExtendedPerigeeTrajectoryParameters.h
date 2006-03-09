#ifndef ExtendedPerigeeTrajectoryParameters_H
#define ExtendedPerigeeTrajectoryParameters_H

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"


/**
 * This class represents simple extention of
 * perigee trajectory parametrization:
 * (rho, theta, phi,tr_im, z_im, mass)
 */

class ExtendedPerigeeTrajectoryParameters
{
public:

ExtendedPerigeeTrajectoryParameters()
{vl = false;}

ExtendedPerigeeTrajectoryParameters(const AlgebraicVector& param,
                                    const TrackCharge& charge)
{
 vl = true;
 par = param;
 ch = charge;
}

/**
 * Access methods
 */		
  bool isValid() const
  {return vl;}
 
  AlgebraicVector vector() const
  {return par;}
 		     
  TrackCharge charge() const
  {return ch;}	

private:

 bool vl;
 AlgebraicVector par;
 TrackCharge ch;
};
#endif
