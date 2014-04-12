#include "RecoVertex/VertexTools/interface/lms_3d.h"
#include "RecoVertex/VertexTools/interface/Lms3d.h"

GlobalPoint Lms3d::operator() ( std::vector<GlobalPoint> & values ) const {
  return lms_3d ( values );
}
