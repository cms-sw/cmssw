#include "RecoVertex/VertexTools/interface/hsm_3d.h"
#include "CommonTools/Statistics/interface/hsm_1d.icc"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

#include <iostream>

/// cordinate wise half sample mode in 3d
GlobalPoint hsm_3d ( const std::vector<GlobalPoint> & values )
{
  const int sze = values.size();
  if ( sze == 0 ) {
      throw VertexException("hsm_3d: no values given.");
  };
  std::vector <float> x_vals, y_vals, z_vals;
  x_vals.reserve(sze-1);
  y_vals.reserve(sze-1);
  z_vals.reserve(sze-1);
  for ( std::vector<GlobalPoint>::const_iterator i=values.begin();
      i!=values.end() ; i++ )
  {
    x_vals.push_back( i->x() );
    y_vals.push_back( i->y() );
    z_vals.push_back( i->z() );
  };

  // FIXME isnt necessary, is it?
  /*
  sort ( x_vals.begin(), x_vals.end() );
  sort ( y_vals.begin(), y_vals.end() );
  sort ( z_vals.begin(), z_vals.end() );*/

  GlobalPoint ret ( hsm_1d(x_vals), hsm_1d(y_vals), hsm_1d(z_vals) );
  return ret;
}
