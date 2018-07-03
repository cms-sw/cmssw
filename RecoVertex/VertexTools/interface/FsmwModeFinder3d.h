#ifndef _FsmwModeFinder3d_H_
#define _FsmwModeFinder3d_H_

#include "RecoVertex/VertexTools/interface/ModeFinder3d.h"

/**
 *  
 *  \class FsmwModeFinder3d,
 *  this is a half sample mode finder that works
 *  coordinate wise, in 3d.
 */
class FsmwModeFinder3d : public ModeFinder3d
{
public:
    /**
     *  Constructor
     *  \param fraction the fraction of data points that have to be 
     *  within the interval.
     *  \param weightExponent
     *  The exponent by which the weights are taken into account.
     *  Default is "w^-1", w being the track distance + cutoff (see below).
     *  \param cutoff
     *  weight = ( cutoff + 10000 * distance ) * weightExponent
     *  \param no_weights_above
     *  ignore weights as long as the number of data points is > x
     */
    FsmwModeFinder3d( float fraction = .5, float weightExponent = -2.,
                      float cutoff=10 /* microns */, int no_weights_above = 10 );
    GlobalPoint operator()( const std::vector< PointAndDistance> & ) const override;
    FsmwModeFinder3d* clone() const override;
private:
    float theFraction;
    float theWeightExponent;
    float theCutoff;
    int theNoWeightsAbove;
};

#endif
