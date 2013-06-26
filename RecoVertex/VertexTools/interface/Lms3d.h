#ifndef LMS3D_H
#define LMS3D_H

#include "RecoVertex/VertexTools/interface/ModeFinder3d.h"

/** 
 *
 * \class Lms3d
 * This is a "Least Median of Squares" mode finder that works coordinate-wise
 * on GlobalPoints.
 */
class Lms3d : public ModeFinder3d {
public:
  virtual GlobalPoint operator () ( std::vector<GlobalPoint> & values ) const;
};

#endif
