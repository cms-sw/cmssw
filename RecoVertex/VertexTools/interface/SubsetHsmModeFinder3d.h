#ifndef SubsetHsmModeFinder3d_H
#define SubsetHsmModeFinder3d_H

#include "RecoVertex/VertexTools/interface/ModeFinder3d.h"

/** 
 *  \class HsmModeFinder3d,
 *  this is a half sample mode finder that works
 *  coordinate wise, in 3d; as an preparatory step we filter out
 *  all values whose distance is above a certain threshold
 *  ( the threshold moves if not enough values are
 *  found within the threshold ).
 */
class SubsetHsmModeFinder3d : public ModeFinder3d {
public:
  GlobalPoint operator () ( const std::vector< PointAndDistance> & values ) const override;
  SubsetHsmModeFinder3d * clone() const override;
};

#endif
