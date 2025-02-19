#include "RecoVertex/LinearizationPointFinders/interface/SubsetHSMLinearizationPointFinder.h"
#include "RecoVertex/VertexTools/interface/SubsetHsmModeFinder3d.h"

SubsetHSMLinearizationPointFinder::SubsetHSMLinearizationPointFinder(
    const signed int n_pairs ) :
  CrossingPtBasedLinearizationPointFinder ( SubsetHsmModeFinder3d(), n_pairs )
{ }

SubsetHSMLinearizationPointFinder::SubsetHSMLinearizationPointFinder(
    const RecTracksDistanceMatrix * m, const signed int n_pairs ) :
  CrossingPtBasedLinearizationPointFinder ( m , SubsetHsmModeFinder3d(), n_pairs )
{ }

SubsetHSMLinearizationPointFinder * SubsetHSMLinearizationPointFinder::clone()
  const
{
  return new SubsetHSMLinearizationPointFinder ( * this );
}
