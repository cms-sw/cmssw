
#include "RecoVertex/TertiaryTracksVertexFinder/interface/PvSvFilter.h"

#include "RecoVertex/TertiaryTracksVertexFinder/interface/TransientTrackInVertices.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include <vector>


PvSvFilter::PvSvFilter ( double maxInPv , TransientVertex & thePV ) : 
  maxFractionPv ( maxInPv ) , 
  thePrimaryVertex ( &thePV ) 
{}

bool PvSvFilter::operator() (const TransientVertex & VertexToFilter ) const 
{
  std::vector<reco::TransientTrack> tracksAtSv = VertexToFilter.originalTracks() ;
  int multSv = tracksAtSv.size() ;
  int inPv = 0 ;
  
  for ( std::vector<reco::TransientTrack>::const_iterator itSvT = tracksAtSv.begin(); 
    itSvT != tracksAtSv.end() ; itSvT++ ) {
    if ( TransientTrackInVertices::isInVertex ( *itSvT , *thePrimaryVertex ) ) inPv++ ;
    else {if(debug) std::cout<<"NOT IN VTX\n";}
  }
  
  double  fracInPv = 0.0 ;
  if ( multSv > 0 ) fracInPv = double ( inPv ) / double ( multSv ) ; 

  if(debug) std::cout <<"[PvSvFilter] frac,max: "<<fracInPv<<","<<maxFractionPv<<std::endl;

  return ( fracInPv < maxFractionPv ) ; 
}

