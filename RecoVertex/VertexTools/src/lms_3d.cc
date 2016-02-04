#include "CommonTools/Statistics/interface/lms_1d.icc"
#include "RecoVertex/VertexTools/interface/lms_3d.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include <vector>

/// least median of squares in three dimensions,
/// doing every dimension separately
GlobalPoint lms_3d ( std::vector<GlobalPoint> values )
{
  const int sze = values.size();
  if ( sze == 0 ) {
      throw VertexException("lms_3d: no values given.");
  };
  std::vector <float> x_vals, y_vals, z_vals;
  x_vals.reserve(sze-1);
  y_vals.reserve(sze-1);
  z_vals.reserve(sze-1);
  for (std:: vector<GlobalPoint>::iterator i=values.begin();
      i!=values.end() ; i++ ) {
    x_vals.push_back( i->x() );
    y_vals.push_back( i->y() );
    z_vals.push_back( i->z() );
  };
  return GlobalPoint ( lms_1d(x_vals), lms_1d(y_vals), lms_1d(z_vals) );
}
