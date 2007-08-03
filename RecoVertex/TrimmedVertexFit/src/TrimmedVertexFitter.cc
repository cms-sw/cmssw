#include "RecoVertex/TrimmedVertexFit/interface/TrimmedVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



TrimmedVertexFitter::TrimmedVertexFitter(const edm::ParameterSet & pSet)
{
  theRector.setMaxNbOfVertices(1);
  setPtCut(pSet.getParameter<double>("PtCut"));
}


CachingVertex 
TrimmedVertexFitter::vertex(const std::vector<reco::TransientTrack> & tracks) const
{
  std::vector<TransientVertex> vtces = theRector.vertices ( tracks );
  if (vtces.size() )
  {
    const TransientVertex & rv = *(vtces.begin());
    LinearizedTrackStateFactory lfac;
    VertexTrackFactory vfac; 
       VertexState state ( rv.position(), rv.positionError() );
     vector < RefCountedVertexTrack > vtrks;
    std::vector<reco::TransientTrack> mytrks = rv.originalTracks();
    for ( std::vector<reco::TransientTrack>::const_iterator rt=mytrks.begin(); 
          rt!=mytrks.end() ; ++rt )
    {
      RefCountedLinearizedTrackState lstate =lfac.linearizedTrackState
       ( rv.position(), *rt );
       
      RefCountedVertexTrack vtrk = vfac.vertexTrack ( lstate, state, 1.0 );
      vtrks.push_back ( vtrk );
    };
    return CachingVertex ( rv.position(), rv.positionError(), vtrks, rv.totalChiSquared() );
  };
 throw VertexException("no vertex found");
}

CachingVertex TrimmedVertexFitter::vertex(
    const vector<RefCountedVertexTrack> & tracks) const
{
  cout << "[TrimmedVertexFitter] method not implemented" << endl;
  throw VertexException("not implemented");
  }

CachingVertex TrimmedVertexFitter::vertex(
    const std::vector<reco::TransientTrack> & tracks, const GlobalPoint& linPoint) const
{
  cout << "[TrimmedVertexFitter] method not implemented" << endl;
  throw VertexException("not implemented");
}

CachingVertex TrimmedVertexFitter::vertex(
    const std::vector<reco::TransientTrack> & tracks, const GlobalPoint& priorPos,
    const GlobalError& priorError) const
{
  cout << "[TrimmedVertexFitter] method not implemented" << endl;
  throw VertexException("not implemented");
}

CachingVertex TrimmedVertexFitter::vertex(
    const vector<RefCountedVertexTrack> & tracks, 
	 const GlobalPoint& priorPos,
	 const GlobalError& priorError) const
{
 cout << "[TrimmedVertexFitter] method not implemented" << endl;
  throw VertexException("not implemented");
}

CachingVertex 
TrimmedVertexFitter::vertex(const vector<reco::TransientTrack> & tracks,
			       const reco::BeamSpot& beamSpot) const
{
 cout << "[TrimmedVertexFitter] method not implemented" << endl;
  throw VertexException("not implemented");
}


TrimmedVertexFitter * TrimmedVertexFitter::clone() const
{
  return new TrimmedVertexFitter( * this );
}
 
void TrimmedVertexFitter::setPtCut ( float cut )
{
  ptcut           = cut;
  theRector.setPtCut ( cut );
}

void TrimmedVertexFitter::setTrackCompatibilityCut ( float cut )
{
  theRector.setTrackCompatibilityCut ( cut );
}

void TrimmedVertexFitter::setVertexFitProbabilityCut ( float cut )
{
  theRector.setVertexFitProbabilityCut ( cut );
}

