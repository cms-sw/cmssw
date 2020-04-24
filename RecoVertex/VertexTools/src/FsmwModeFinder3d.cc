#include "CommonTools/Clustering1D/interface/FsmwClusterizer1D.h"
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
}

GlobalPoint FsmwModeFinder3d::operator() (
    const std::vector< PointAndDistance> & values ) const
{
    typedef Cluster1D<void> SimpleCluster;
    std::vector< SimpleCluster > vx, vy, vz;
    vx.reserve ( values.size() - 1 );
    vy.reserve ( values.size() - 1 );
    vz.reserve ( values.size() - 1 );
    std::vector < const void * > emptyvec;

    for ( std::vector< PointAndDistance >::const_iterator i = values.begin();
          i != values.end(); ++i )
    {
        float weight = 1.;
        if ( static_cast<int>( values.size() ) < theNoWeightsAbove )
        {
            // compute weights if we have fewer than theNoWeightsAbove
            // data points
            weight = pow ( theCutoff + 10000 * i->second, theWeightExponent );
        };

        SimpleCluster tmp_x ( Measurement1D ( i->first.x(), 1.0 ),
                              emptyvec, weight );
        SimpleCluster tmp_y ( Measurement1D ( i->first.y(), 1.0 ),
                              emptyvec, weight );
        SimpleCluster tmp_z ( Measurement1D ( i->first.z(), 1.0 ),
                              emptyvec, weight );
        vx.push_back ( tmp_x );
        vy.push_back ( tmp_y );
        vz.push_back ( tmp_z );
    };

    FsmwClusterizer1D<void> algo( theFraction );
    std::vector < SimpleCluster > cresx = algo(vx).first;
    std::vector < SimpleCluster > cresy = algo(vy).first;
    std::vector < SimpleCluster > cresz = algo(vz).first;
    assert ( !cresx.empty() && !cresy.empty() && !cresz.empty() );

    GlobalPoint ret ( cresx.begin()->position().value(),
                      cresy.begin()->position().value(),
                      cresz.begin()->position().value() );
    return ret;
}

FsmwModeFinder3d * FsmwModeFinder3d::clone() const
{
    return new FsmwModeFinder3d( *this );
}


