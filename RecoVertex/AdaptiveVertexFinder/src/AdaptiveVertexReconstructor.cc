#include "RecoVertex/AdaptiveVertexFinder/interface/AdaptiveVertexReconstructor.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoVertex/VertexTools/interface/DummyVertexSmoother.h"
#include <algorithm>

using namespace std;

void AdaptiveVertexReconstructor::erase (
    const TransientVertex & newvtx,
    std::set < reco::TransientTrack > & remainingtrks ) const
{
  /*
   * Erase tracks that are in newvtx from remainingtrks */
  const std::vector < reco::TransientTrack > & origtrks = newvtx.originalTracks();
  bool erased=false;
  for ( std::vector< reco::TransientTrack >::const_iterator i=origtrks.begin();
        i!=origtrks.end(); ++i )
  {
    double weight = newvtx.trackWeight ( *i );
    if ( weight > theMinWeight )
    {
      remainingtrks.erase ( *i );
      erased=true;
    };
  };
}

AdaptiveVertexReconstructor::AdaptiveVertexReconstructor(
    float primcut, float seccut, float min_weight ) : thePrimCut ( primcut ),
       theSecCut ( seccut ), theMinWeight ( min_weight )
{}

AdaptiveVertexReconstructor::AdaptiveVertexReconstructor( const edm::ParameterSet & m )
  : thePrimCut ( 2.0 ), theSecCut ( 6.0 ), theMinWeight ( 0.5 )
{
  try {
    thePrimCut =  m.getParameter<double>("primcut");
    theSecCut  =  m.getParameter<double>("seccut");
    theMinWeight = m.getParameter<double>("minweight");
  } catch ( edm::Exception & e ) {
    edm::LogError ("") << e.what();
  }
}

TransientVertex AdaptiveVertexReconstructor::cleanUp ( const TransientVertex & old ) const
{
  vector < reco::TransientTrack > trks = old.originalTracks();
  vector < reco::TransientTrack > newtrks;
  TransientVertex::TransientTrackToFloatMap mp;
  for ( vector< reco::TransientTrack >::const_iterator i=trks.begin();
        i!=trks.end() ; ++i )
  {
    if ( old.trackWeight ( *i ) > 1.e-8 )
    {
      newtrks.push_back ( *i );
      mp[*i]=old.trackWeight ( *i );
    }
  }

  TransientVertex ret;

  if ( old.hasPrior() )
  {
    VertexState priorstate ( old.priorPosition(), old.priorError() );
    ret=TransientVertex ( priorstate, old.vertexState(), newtrks,
        old.totalChiSquared(), old.degreesOfFreedom() );
  } else {
    ret=TransientVertex ( old.vertexState(), newtrks,
                          old.totalChiSquared(), old.degreesOfFreedom() );
  }
  ret.weightMap ( mp ); // set weight map
  vector < reco::TransientTrack > newrfs;
  vector < reco::TransientTrack > oldrfs=old.refittedTracks();
  for ( vector< reco::TransientTrack >::const_iterator i=oldrfs.begin(); i!=oldrfs.end() ; ++i )
  {
    if ( old.trackWeight ( old.originalTrack ( *i ) ) > 1.e-8 )
    {
      newrfs.push_back ( *i );
    }
  }
  ret.refittedTracks ( newrfs ); // copy refitted tracks
  return ret;
}
  
std::vector<TransientVertex> 
    AdaptiveVertexReconstructor::vertices(const std::vector<reco::TransientTrack> & t, 
        const reco::BeamSpot & s ) const
{
  return vertices ( t,s, true );
}

std::vector<TransientVertex> AdaptiveVertexReconstructor::vertices (
    const std::vector<reco::TransientTrack> & tracks ) const
{
  return vertices ( tracks, reco::BeamSpot() , false );
}

std::vector<TransientVertex> AdaptiveVertexReconstructor::vertices (
    const std::vector<reco::TransientTrack> & tracks,
    const reco::BeamSpot & s, bool usespot ) const
{
  std::vector < TransientVertex > ret;
  std::set < reco::TransientTrack > remainingtrks;

  std::copy(tracks.begin(), tracks.end(), 
	    std::inserter(remainingtrks, remainingtrks.begin()));

  int ctr=0;
  unsigned int n_tracks = remainingtrks.size();
  try {
    while ( remainingtrks.size() > 1 )
    {
      ctr++;
      float cut = theSecCut;
      if ( ret.size() == 0 )
      {
        cut = thePrimCut;
      };
      GeometricAnnealing ann ( cut );
      AdaptiveVertexFitter fitter ( ann, DefaultLinearizationPointFinder(),
          KalmanVertexUpdator(), KalmanVertexTrackCompatibilityEstimator(),
          DummyVertexSmoother() );
      vector < reco::TransientTrack > fittrks;
      fittrks.reserve ( remainingtrks.size() );

      std::copy(remainingtrks.begin(), remainingtrks.end(), std::back_inserter(fittrks));

      TransientVertex tmpvtx;
      if ( ret.size() == 0 && usespot )
      {
        tmpvtx=fitter.vertex ( fittrks, s );
      } else {
        tmpvtx=fitter.vertex ( fittrks );
      }
      TransientVertex newvtx = cleanUp ( tmpvtx );
      ret.push_back ( newvtx );
      erase ( newvtx, remainingtrks );
      if ( n_tracks == remainingtrks.size() )
      {
        edm::LogWarning("") << "all tracks (" << n_tracks
             << ") would be recycled for next fit."
             << " breaking after reconstruction of "
             << ret.size() << " vertices.";
        break;
      };
      n_tracks = remainingtrks.size();
    };
  } catch ( exception & e ) {
    // Will catch all (not enough significant tracks exceptions.
    // in this case, the iteration can safely terminate.
    // cout << "[AdaptiveVertexReconstructor] exception: " << e.what() << endl;
  } catch ( ... ) {
    // cout << "[AdaptiveVertexReconstructor] exception" << endl;
  };
  return ret;
}
