
#include "RecoVertex/TertiaryTracksVertexFinder/interface/V0SvFilter.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/TertiaryTracksVertexFinder/interface/VertexMass.h"

V0SvFilter::V0SvFilter(double massWindow) : theMassWindow(massWindow) ,
  theK0sMass(0.4976)
 {}

bool V0SvFilter::operator()(const TransientVertex& vtx) const
{
  VertexMass theVertexMass;

  // only filter if exactly 2 tracks
  if (vtx.originalTracks().size() != 2) return true;   

   // filter only if they have opposite charge
  if ( (vtx.originalTracks()[0].charge() * vtx.originalTracks()[1].charge()) 
       != -1 ) return true ;

  // filter on mass window
  return (fabs(theVertexMass(vtx)-theK0sMass) > theMassWindow);
}
