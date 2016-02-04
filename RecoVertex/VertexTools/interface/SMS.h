#ifndef SMS_H
#define SMS_H

#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <iostream>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

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

  GlobalPoint location ( const std::vector < GlobalPoint > & ) const;
  GlobalPoint location ( const std::vector < std::pair < GlobalPoint, float > > & ) const;

private:
  SMSType theType;
  float theRatio;

};

#endif /* def SMS */
