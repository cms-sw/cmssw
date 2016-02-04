#include "RecoVertex/LinearizationPointFinders/interface/FsmwLinearizationPointFinder.h"
#include "RecoVertex/VertexTools/interface/FsmwModeFinder3d.h"

FsmwLinearizationPointFinder::FsmwLinearizationPointFinder(
    signed int n_pairs, float we, float frac, float cut, int nwa ) :
  CrossingPtBasedLinearizationPointFinder ( FsmwModeFinder3d(frac,we,cut,nwa), n_pairs )
{ }

FsmwLinearizationPointFinder::FsmwLinearizationPointFinder(
    const RecTracksDistanceMatrix * m, signed int n_pairs, float we, 
       float frac, float cut, int nwa ) :
  CrossingPtBasedLinearizationPointFinder ( m , 
      FsmwModeFinder3d( frac,we,cut,nwa ), n_pairs )
{ }

FsmwLinearizationPointFinder * FsmwLinearizationPointFinder::clone()
  const
{
  return new FsmwLinearizationPointFinder ( * this );
}
