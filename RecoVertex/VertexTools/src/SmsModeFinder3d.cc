#include "RecoVertex/VertexTools/interface/SmsModeFinder3d.h"

SmsModeFinder3d::SmsModeFinder3d ( const SMS & algo ) :
    theAlgo(algo)
{}

GlobalPoint SmsModeFinder3d::operator() ( const std::vector<PointAndDistance> & values ) const
{
  std::vector < std::pair < GlobalPoint, float > > weighted;
  for ( std::vector< PointAndDistance >::const_iterator i=values.begin(); 
        i!=values.end() ; ++i )
  {
    float weight = pow ( 10 + 10000 * i->second, -2 );
    weighted.push_back ( std::pair < GlobalPoint, float > ( i->first, weight ) );
  };
  return theAlgo.location( weighted );
}

SmsModeFinder3d * SmsModeFinder3d::clone() const
{
  return new SmsModeFinder3d ( * this );
}
