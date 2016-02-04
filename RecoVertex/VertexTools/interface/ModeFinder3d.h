#ifndef ModeFinder3d_H
#define ModeFinder3d_H

#include <functional>
#include <vector>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

/**  \class ModeFinder3d
 *
 *   A ModeFinder returns a GlobalPoint, given a vector of ( GlobalPoint plus
 *   weight ). [ weight := distance of the points of closest Approach ].
 */

class ModeFinder3d : public std::unary_function <
   std::vector< std::pair < GlobalPoint , float > > , GlobalPoint > {
public:
  typedef std::pair < GlobalPoint, float > PointAndDistance;
  virtual GlobalPoint operator () ( const std::vector< PointAndDistance > & ) const=0;

  virtual ~ModeFinder3d() {};
  virtual ModeFinder3d * clone () const =0;
};

#endif
