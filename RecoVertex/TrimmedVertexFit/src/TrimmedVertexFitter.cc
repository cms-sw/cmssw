#include "RecoVertex/TrimmedVertexFit/interface/TrimmedVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



TrimmedVertexFitter::TrimmedVertexFitter()
{
  theRector.setMaxNbOfVertices(1);
  setPtCut(0.);
}

TrimmedVertexFitter::TrimmedVertexFitter(const edm::ParameterSet & pSet)
{
  theRector.setMaxNbOfVertices(1);
  setPtCut(pSet.getParameter<double>("minPt"));
  setTrackCompatibilityCut ( pSet.getParameter<double>("trackCompatibilityCut") );
  setVertexFitProbabilityCut ( pSet.getParameter<double>("vtxFitProbCut") );
}

CachingVertex<5> 
TrimmedVertexFitter::vertex(const std::vector<reco::TransientTrack> & tracks) const
{
  std::vector<TransientVertex> vtces = theRector.vertices ( tracks );
  if (!vtces.empty() )
  {
    const TransientVertex & rv = *(vtces.begin());
    LinearizedTrackStateFactory lfac;
    VertexTrackFactory<5> vfac; 
       VertexState state ( rv.position(), rv.positionError() );
     std::vector < RefCountedVertexTrack > vtrks;
    std::vector<reco::TransientTrack> mytrks = rv.originalTracks();
    for ( std::vector<reco::TransientTrack>::const_iterator rt=mytrks.begin(); 
          rt!=mytrks.end() ; ++rt )
    {
      RefCountedLinearizedTrackState lstate =lfac.linearizedTrackState
       ( rv.position(), *rt );
       
      RefCountedVertexTrack vtrk = vfac.vertexTrack ( lstate, state, 1.0 );
      vtrks.push_back ( vtrk );
    };
    return CachingVertex<5> ( rv.position(), rv.positionError(), vtrks, rv.totalChiSquared() );
  };
  return CachingVertex<5>();
}

CachingVertex<5> TrimmedVertexFitter::vertex(
    const std::vector<RefCountedVertexTrack> & tracks) const
{
  std::cout << "[TrimmedVertexFitter] method not implemented" << std::endl;
  throw VertexException("not implemented");
}

CachingVertex<5> TrimmedVertexFitter::vertex(
    const std::vector<RefCountedVertexTrack> & tracks,
    const reco::BeamSpot & spot ) const
{
  std::cout << "[TrimmedVertexFitter] method not implemented" << std::endl;
  throw VertexException("not implemented");
}


CachingVertex<5> TrimmedVertexFitter::vertex(
    const std::vector<reco::TransientTrack> & tracks, const GlobalPoint& linPoint) const
{
  std::cout << "[TrimmedVertexFitter] method not implemented" << std::endl;
  throw VertexException("not implemented");
}

CachingVertex<5> TrimmedVertexFitter::vertex(
    const std::vector<reco::TransientTrack> & tracks, const GlobalPoint& priorPos,
    const GlobalError& priorError) const
{
  std::cout << "[TrimmedVertexFitter] method not implemented" << std::endl;
  throw VertexException("not implemented");
}

CachingVertex<5> TrimmedVertexFitter::vertex(
    const std::vector<RefCountedVertexTrack> & tracks, 
	 const GlobalPoint& priorPos,
	 const GlobalError& priorError) const
{
 std::cout << "[TrimmedVertexFitter] method not implemented" << std::endl;
  throw VertexException("not implemented");
}

CachingVertex<5> 
TrimmedVertexFitter::vertex(const std::vector<reco::TransientTrack> & tracks,
			       const reco::BeamSpot& beamSpot) const
{
 std::cout << "[TrimmedVertexFitter] method not implemented" << std::endl;
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

