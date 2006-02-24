#ifndef SMS_H
#define SMS_H

#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <iostream>
#include "Geometry/Vector/interface/GlobalPoint.h"

using namespace std;

/**
 * Class to compute the SMS location estimator The SMS estimator is the mean
 * value of a set of observations with Small Median of Squared distances.
 */

class SMS
{
public:
  enum SMSType { None        = 0,
                 Interpolate = 1,
                 Iterate     = 2,
                 Weighted    = 4 };
  /**
   *  Constructor.
   *  \param tp What specific kind of SMS algorithm do you want?
   *  \param q  What fraction of data points are considered for the
   *  "next step"?
   */
  SMS ( SMSType tp = (SMSType) (Interpolate | Iterate | Weighted), float q=0.5 );

  GlobalPoint location ( const vector < GlobalPoint > & ) const;
  GlobalPoint location ( const vector < pair < GlobalPoint, float > > & ) const;

private:
  SMSType theType;
  float theRatio;

};

#endif /* def SMS */
