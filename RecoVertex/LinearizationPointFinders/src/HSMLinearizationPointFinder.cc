#include "RecoVertex/LinearizationPointFinders/interface/HSMLinearizationPointFinder.h"
#include "RecoVertex/VertexTools/interface/HsmModeFinder3d.h"

HSMLinearizationPointFinder::HSMLinearizationPointFinder(
    const signed int n_pairs ) :
  CrossingPtBasedLinearizationPointFinder ( HsmModeFinder3d(), n_pairs )
{ }

HSMLinearizationPointFinder::HSMLinearizationPointFinder(
    const RecTracksDistanceMatrix * m, const signed int n_pairs ) :
  CrossingPtBasedLinearizationPointFinder ( m , HsmModeFinder3d(), n_pairs )
{ }

HSMLinearizationPointFinder * HSMLinearizationPointFinder::clone()
  const
{
  return new HSMLinearizationPointFinder ( * this );
}
