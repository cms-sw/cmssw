#include "RecoVertex/VertexTools/interface/hsm_3d.h"
#include "RecoVertex/VertexTools/interface/HsmModeFinder3d.h"

/** Half sample mode in 3d, as a functional class.
 */

GlobalPoint HsmModeFinder3d::operator() ( const vector< PointAndDistance> & values ) const
{
  vector < GlobalPoint > v;
  for ( vector< PointAndDistance >::const_iterator i=values.begin(); 
      i!=values.end() ; ++i ) 
  {
    v.push_back ( i->first );
  };
  return hsm_3d ( v );
};

HsmModeFinder3d * HsmModeFinder3d::clone() const
{
  return new HsmModeFinder3d ( * this );
};
