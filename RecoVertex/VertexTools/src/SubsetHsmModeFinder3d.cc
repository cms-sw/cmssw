#include "RecoVertex/VertexTools/interface/hsm_3d.h"
#include "RecoVertex/VertexTools/interface/SubsetHsmModeFinder3d.h"

#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

#include <algorithm>

typedef std::pair < GlobalPoint, float > PointAndDistance;

namespace {
  struct compareByDistance
  {
    bool operator() ( const PointAndDistance & p1,
        const PointAndDistance & p2 ) {
      return ( p1.second < p2.second );
    };
  };
}

GlobalPoint SubsetHsmModeFinder3d::operator() ( const std::vector< PointAndDistance> & values )
    const 
{
  if ( values.empty() )
  {
    throw VertexException ("SubsetHsmModeFinder3d: no value given.");
  };

  std::vector < GlobalPoint > pts; pts.reserve ( values.size()-1 );
  std::vector< PointAndDistance> sorted_values ( values.size() );
  partial_sort_copy ( values.begin(), values.end(),
      sorted_values.begin(), sorted_values.end(), compareByDistance() );

  std::vector< PointAndDistance>::iterator end = sorted_values.end();
  std::vector< PointAndDistance>::iterator begin = sorted_values.begin();

  float dmax = 0.004; // 40 microns, as a first try.

  // we want at least 30 values
  unsigned int min_num = values.size() < 30 ? values.size() : 30;

  // we also want at least 50 % of all values
  if ( values.size() > 2 * min_num ) min_num = (int) values.size() / 2;

  while ( pts.size() < min_num )
  {
    // we cut at a dmax
    std::vector< PointAndDistance>::iterator i;
    for ( i=begin; i!=end && ( i->second < dmax )  ; ++i )
    {
      pts.push_back ( i->first );
    };
    dmax +=0.003; // add 30 microns with every iteration
    begin=i;
  };

  GlobalPoint ret = hsm_3d ( pts );
  return ret;
}

SubsetHsmModeFinder3d * SubsetHsmModeFinder3d::clone() const
{
  return new SubsetHsmModeFinder3d ( * this );
}
