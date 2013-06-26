#ifndef LMS3D_H
#define LMS3D_H

#include "RecoVertex/VertexTools/interface/ModeFinder3d.h"

/** Least Median sum of squares mode finder. Works coordinate wise.
 */
class LmsModeFinder3d : public ModeFinder3d {
public:
  virtual GlobalPoint operator () ( const std::vector< PointAndDistance> & values ) const;
  virtual LmsModeFinder3d * clone() const;
};

#endif
