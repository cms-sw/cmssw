#ifndef LMS3D_H
#define LMS3D_H

#include "RecoVertex/VertexTools/interface/ModeFinder3d.h"
#include "RecoVertex/VertexTools/interface/SMS.h"

/** Least Median sum of squares mode finder. Works coordinate wise.
 */
class SmsModeFinder3d : public ModeFinder3d {
public:
  SmsModeFinder3d ( const SMS & algo = SMS() );
  GlobalPoint operator () ( const std::vector< PointAndDistance> & values ) const override;
  SmsModeFinder3d * clone() const override;
private:
  SMS theAlgo;
};

#endif
