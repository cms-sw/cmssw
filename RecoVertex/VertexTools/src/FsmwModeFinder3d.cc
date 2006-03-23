#include "CommonReco/Clustering1D/interface/FsmwClusterizer1D.h"
#include "RecoVertex/VertexTools/interface/FsmwModeFinder3d.h"

#include <cmath>
#include <cassert>

/** Half sample mode in 3d, as a functional class.
 */

FsmwModeFinder3d::FsmwModeFinder3d( float fraction, float weightExp, 
    float cutoff, int no_w_a ) : theFraction ( fraction ), 
    theWeightExponent ( weightExp ), theCutoff(cutoff), 
    theNoWeightsAbove ( no_w_a )
{
  assert ( theFraction > 0. && theFraction < 1. );
};

GlobalPoint FsmwModeFinder3d::operator() ( 
    const vector< PointAndDistance> & values ) const
{
  typedef Cluster1D<void> Cluster;
  vector < Cluster > vx, vy, vz;
  vx.reserve ( values.size() - 1 );
  vy.reserve ( values.size() - 1 );
  vz.reserve ( values.size() - 1 );
  vector < const void * > emptyvec;

  for ( vector< PointAndDistance >::const_iterator i=values.begin();
      i!=values.end() ; ++i )
  {
    float weight = 1.;
    if ( values.size() < theNoWeightsAbove )
    {
      // compute weights if we have fewer than theNoWeightsAbove
      // data points
      weight = pow ( theCutoff + 10000 * i->second, theWeightExponent );
    };

    Cluster tmp_x ( Measurement1D ( i->first.x(), 1.0 ),
                    emptyvec, weight );
    Cluster tmp_y ( Measurement1D ( i->first.y(), 1.0 ),
                    emptyvec, weight );
    Cluster tmp_z ( Measurement1D ( i->first.z(), 1.0 ),
                    emptyvec, weight );
    vx.push_back ( tmp_x );
    vy.push_back ( tmp_y );
    vz.push_back ( tmp_z );
  };
  FsmwClusterizer1D<void> algo( theFraction );
  vector < Cluster > cresx = algo(vx).first;
  vector < Cluster > cresy = algo(vy).first;
  vector < Cluster > cresz = algo(vz).first;
  assert ( cresx.size() && cresy.size() && cresz.size() );

  GlobalPoint ret ( cresx.begin()->position().value(),
                    cresy.begin()->position().value(),
                    cresz.begin()->position().value() );
  return ret;
};

FsmwModeFinder3d * FsmwModeFinder3d::clone() const
{
  return new FsmwModeFinder3d ( * this );
};
