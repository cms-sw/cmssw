#ifndef LMS3D_H
#define LMS3D_H

#include "RecoVertex/VertexTools/interface/ModeFinder3d.h"

/** Least Median sum of squares mode finder. Works coordinate wise.
 */
class LmsModeFinder3d : public ModeFinder3d {
public:
  GlobalPoint operator () ( const std::vector< PointAndDistance> & values ) const override;
  LmsModeFinder3d * clone() const override;
};

#endif
