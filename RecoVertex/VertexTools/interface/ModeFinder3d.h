#ifndef ModeFinder3d_H
#define ModeFinder3d_H

#include <functional>
#include <vector>
#include "Geometry/Vector/interface/GlobalPoint.h"

using namespace std;

/**  \class ModeFinder3d
 *
 *   A ModeFinder returns a GlobalPoint, given a vector of ( GlobalPoint plus
 *   weight ). [ weight := distance of the points of closest Approach ].
 */

class ModeFinder3d : public unary_function <
   vector< pair < GlobalPoint , float > > , GlobalPoint > {
public:
  typedef pair < GlobalPoint, float > PointAndDistance;
  virtual GlobalPoint operator () ( const vector< PointAndDistance > & ) const=0;
  virtual ModeFinder3d * clone () const =0;
};

#endif
