#include "RecoVertex/LinearizationPointFinders/interface/LMSLinearizationPointFinder.h"
#include "RecoVertex/VertexTools/interface/LmsModeFinder3d.h"

LMSLinearizationPointFinder::LMSLinearizationPointFinder(
    const signed int n_pairs ) :
  CrossingPtBasedLinearizationPointFinder ( LmsModeFinder3d(), n_pairs )
{ }

LMSLinearizationPointFinder::LMSLinearizationPointFinder(
    const RecTracksDistanceMatrix * m, const signed int n_pairs ) :
  CrossingPtBasedLinearizationPointFinder ( m , LmsModeFinder3d(), n_pairs )
{ }

LMSLinearizationPointFinder * LMSLinearizationPointFinder::clone()
  const
{
  return new LMSLinearizationPointFinder ( * this );
}
