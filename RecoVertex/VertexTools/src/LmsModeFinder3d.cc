#include "RecoVertex/VertexTools/interface/lms_3d.h"
#include "RecoVertex/VertexTools/interface/LmsModeFinder3d.h"

GlobalPoint LmsModeFinder3d::operator() ( const vector<PointAndDistance> & values ) const {
  vector < GlobalPoint > v;
  for ( vector< PointAndDistance >::const_iterator i=values.begin(); 
      i!=values.end() ; ++i ) 
  {
    v.push_back ( i->first );
  };
  return lms_3d ( v );
};

LmsModeFinder3d * LmsModeFinder3d::clone() const
{
  return new LmsModeFinder3d ( * this );
};
