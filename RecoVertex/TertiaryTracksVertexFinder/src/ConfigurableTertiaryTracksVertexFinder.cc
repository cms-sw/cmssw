
#include "RecoVertex/TertiaryTracksVertexFinder/interface/ConfigurableTertiaryTracksVertexFinder.h"

#include "RecoVertex/TertiaryTracksVertexFinder/interface/VertexMass.h"
#include "RecoVertex/TertiaryTracksVertexFinder/interface/V0SvFilter.h"
#include "RecoVertex/TertiaryTracksVertexFinder/interface/Flight2DSvFilter.h"
#include "RecoVertex/TertiaryTracksVertexFinder/interface/PvSvFilter.h"
#include "RecoVertex/TertiaryTracksVertexFinder/interface/TransientTrackInVertices.h"

using namespace reco;
using namespace std;

//-----------------------------------------------------------------------------
// constructor

ConfigurableTertiaryTracksVertexFinder::ConfigurableTertiaryTracksVertexFinder(
  const VertexFitter<5> * vf, 
  const VertexUpdator<5> * vu, const VertexTrackCompatibilityEstimator<5> * ve)
{
  theTKVF = new ConfigurableTrimmedVertexFinder(vf,vu,ve);

  theMinTrackPt = 1.0;
  theMaxVtxMass = 6.5;
  theMaxSigOnDistTrackToB = 3.0; // this is being overwritten to 10 in AddTvTtrack (why?)

  theMaxInPvFrac = 0.65;

  // set up V0SvFilter
  theK0sMassWindow = 0.05; // mass window around K0s
  theV0SvFilter = new V0SvFilter(theK0sMassWindow);

  // set up Flight2DSvFilter
  theMaxDist2D = 2.5;  // max transv. dist to beam line
  theMinDist2D = 0.01; // min transv. dist to beam line
  theMinSign2D = 3.0;  // min transverse distance significance
  theMinTracks = 2;    // min number of tracks
  theFlight2DSvFilter= new Flight2DSvFilter(theMaxDist2D,theMinDist2D,
					    theMinSign2D,theMinTracks);

  //  thePrimaryVertex = new TransientVertex;
  // FIXME this is incomplete!? -> get real primary vertex!

  //theNewTrackInfoVector = new NewTrackInfoVector;

}

//-----------------------------------------------------------------------------
// destructor

ConfigurableTertiaryTracksVertexFinder::~ConfigurableTertiaryTracksVertexFinder()
{
  delete theTKVF;
  delete theV0SvFilter;
  delete theFlight2DSvFilter;
}

//-----------------------------------------------------------------------------

std::vector<TransientVertex> ConfigurableTertiaryTracksVertexFinder::vertices(
									      const std::vector<reco::TransientTrack> & tracks, const TransientVertex& pv) const
{
  // there is a PV
  if (pv.isValid()) {
    theFlight2DSvFilter->setPrimaryVertex( pv);
    if (debug) cout <<"[TTVF] calling with a real PV ...\n";
    return this->reconstruct(tracks,pv);
  }
  else return this->vertices(tracks);

}

//-----------------------------------------------------------------------------

std::vector<TransientVertex> ConfigurableTertiaryTracksVertexFinder::vertices(
  const std::vector<reco::TransientTrack> & tracks) const 
{
  // there is no PV
  theFlight2DSvFilter->useNotPV();
  TransientVertex dummyVertex;
  if (debug) cout <<"[TTVF] calling with a dummy PV ...\n";
  return this->reconstruct(tracks,dummyVertex);

}

//-----------------------------------------------------------------------------

std::vector<TransientVertex> ConfigurableTertiaryTracksVertexFinder::reconstruct(
										 const std::vector<reco::TransientTrack> & tracks, const TransientVertex& pv) const
{
  // get  primary vertices;
  std::vector<TransientVertex> primaryVertices;
  if(pv.isValid()) { primaryVertices.push_back(pv); 
    if (debug) cout <<"[TTVF] add PV ...\n";
  }

  VertexMass theVertexMass;

  //filter tracks in pt
  std::vector<reco::TransientTrack> filteredTracks;
  for (std::vector<reco::TransientTrack>::const_iterator it=tracks.begin();
       it!=tracks.end();it++)
    if ((*it).impactPointState().globalMomentum().perp() > theMinTrackPt) 
      filteredTracks.push_back(*it);

  if (debug) cout <<"[TTVF] tracks: " << filteredTracks.size() <<endl;

  // get vertices
  std::vector<TransientVertex> vertices;
  if (filteredTracks.size()>1) vertices = theTKVF->vertices(filteredTracks);

  if (debug) cout <<"[TTVF] found secondary vertices with TKVF: "<<vertices.size()<<endl;

  std::vector<TransientVertex> secondaryVertices;

  for(std::vector<TransientVertex>::const_iterator ivx=vertices.begin();
      ivx!=vertices.end(); ivx++) {
    TransientVertex vtx=*ivx;

    double mass=theVertexMass(vtx);
    if (debug) cout <<"[TTVF] new svx: mass: "<<mass<<endl;

    if ((*theV0SvFilter)(vtx)) {
      if (debug) cout <<"[TTVF] survived V0SvFilter\n";
      if((*theFlight2DSvFilter)(vtx)) {
	if (debug) cout <<"[TTVF] survived 2DSvFilter\n";
	if (mass<theMaxVtxMass) {
	  if (!primaryVertices.empty()) {
	    PvSvFilter thePvSvFilter(theMaxInPvFrac,primaryVertices[0]);
	    if (thePvSvFilter(vtx)) secondaryVertices.push_back(vtx);
	    else { if (debug) cout <<"[TTVF] failed PvSvFilter\n";}
	  }
	  else secondaryVertices.push_back(vtx);
	}
	else {if(debug)cout<<"[TTVF] failed mass cut\n";}
      }
    }

  }

  if (debug) cout<<"[TTVF] remaining svx: "<<secondaryVertices.size()<<endl;

  if (primaryVertices.empty() || secondaryVertices.empty()) 
    return secondaryVertices; // unable to reconstruct b-flight-trajectory

  if (debug) cout<<"[TTVF] still here ...\n";

  // find tracks not used in primaryVertex or in vertices
  vector<TransientTrack> unusedTracks;
  for( vector<TransientTrack>::const_iterator itT = filteredTracks.begin(); 
    itT != filteredTracks.end(); itT++ ) 
    if( (!TransientTrackInVertices::isInVertex((*itT),primaryVertices)) 
     && (!TransientTrackInVertices::isInVertex((*itT),vertices)) )
       unusedTracks.push_back( *itT ); 
  if (debug) cout <<"[TTVF] remaining tracks: "<<unusedTracks.size()<<endl;

  // now add tracks to the SV candidate
  AddTvTrack MyAddTVTrack( &primaryVertices, &secondaryVertices, 
			   theMaxSigOnDistTrackToB);  
  vector<TransientVertex> newVertices =
    MyAddTVTrack.getSecondaryVertices(unusedTracks); 

  // for tdr studies
  theTrackInfoVector = MyAddTVTrack.getTrackInfo();

  //std::vector<pair<reco::TransientTrack,double> > theTrackInfo;
  //std::vector<pair<reco::TransientTrack,double* > > theTrackInfo2;
  //theTrackInfo = MyAddTVTrack.getTrackInfo();
  //theTrackInfo2= MyAddTVTrack.getTrackInfo2();

  //TrackInfo = theTrackInfo;
  //TrackInfo2= theTrackInfo2;

  if (debug) cout <<"[TTVF] vertices found: "<<newVertices.size()<<endl;
  return newVertices;

}



