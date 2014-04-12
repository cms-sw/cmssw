#ifndef HsmModeFinder3d_H
#define HsmModeFinder3d_H

#include "RecoVertex/VertexTools/interface/ModeFinder3d.h"

/** 
 *  
 *  \class HsmModeFinder3d,
 *  this is a half sample mode finder that works
 *  coordinate wise, in 3d.
 */
class HsmModeFinder3d : public ModeFinder3d {
public:
  virtual GlobalPoint operator () ( const std::vector< PointAndDistance> & ) const;
  virtual HsmModeFinder3d * clone() const;
};

#endif
