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
    set < reco::TransientTrack > & remainingtrks ) const
{
  /*
   * Erase tracks that are in newvtx from remainingtrks */
  const vector < reco::TransientTrack > & origtrks = newvtx.originalTracks();
  bool erased=false;
  for ( vector< reco::TransientTrack >::const_iterator i=origtrks.begin();
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
       theSecCut ( seccut ), theMinWeight ( min_weight ),
       theWeightThreshold ( 0.001 )
{}

AdaptiveVertexReconstructor::AdaptiveVertexReconstructor( const edm::ParameterSet & m )
  : thePrimCut ( 2.0 ), theSecCut ( 6.0 ), theMinWeight ( 0.5 ),
  theWeightThreshold ( 0.001 )
{
  try {
    thePrimCut =  m.getParameter<double>("primcut");
    theSecCut  =  m.getParameter<double>("seccut");
    theMinWeight = m.getParameter<double>("minweight");
    theWeightThreshold = m.getParameter<double>("weightthreshold");
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
  if ( !newrfs.empty() ) ret.refittedTracks ( newrfs ); // copy refitted tracks
  return ret;
}
  
vector<TransientVertex> 
    AdaptiveVertexReconstructor::vertices(const vector<reco::TransientTrack> & t, 
        const reco::BeamSpot & s ) const
{
  return vertices ( t,s, true );
}

vector<TransientVertex> AdaptiveVertexReconstructor::vertices (
    const vector<reco::TransientTrack> & tracks ) const
{
  return vertices ( tracks, reco::BeamSpot() , false );
}

vector<TransientVertex> AdaptiveVertexReconstructor::vertices (
    const vector<reco::TransientTrack> & tracks,
    const reco::BeamSpot & s, bool usespot ) const
{
  vector < TransientVertex > ret;
  set < reco::TransientTrack > remainingtrks;

  copy(tracks.begin(), tracks.end(), 
	    inserter(remainingtrks, remainingtrks.begin()));

  int ctr=0;
  unsigned int n_tracks = remainingtrks.size();

  // cout << "[AdaptiveVertexReconstructor] DEBUG ::vertices!!" << endl;
  try {
    while ( remainingtrks.size() > 1 )
    {
      /*
      cout << "[AdaptiveVertexReconstructor] next round: "
           << remainingtrks.size() << endl;
           */
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
      fitter.setWeightThreshold ( 0. ); // need to set it or else we have 
      // unwanted exceptions to deal with.
      // cleanup can come later!
      vector < reco::TransientTrack > fittrks;
      fittrks.reserve ( remainingtrks.size() );

      copy(remainingtrks.begin(), remainingtrks.end(), back_inserter(fittrks));

      TransientVertex tmpvtx;
      if ( (ret.size() == 0) && usespot )
      {
        // cout << "[AdaptiveVertexReconstructor] fitting w/ beamspot" << endl;
        tmpvtx=fitter.vertex ( fittrks, s );
      } else {
        // cout << "[AdaptiveVertexReconstructor] fitting w/o beamspot" << endl;
        tmpvtx=fitter.vertex ( fittrks );
      }
      TransientVertex newvtx = cleanUp ( tmpvtx );
      ret.push_back ( newvtx );
      erase ( newvtx, remainingtrks );
      if ( n_tracks == remainingtrks.size() )
      {
        if ( usespot )
        {
          // try once more without beamspot constraint!
          usespot=false;
          edm::LogWarning("AdaptiveVertexReconstructor") 
            << "no tracks in vertex. trying again without beamspot constraint!";
          continue;
        }
        edm::LogWarning("AdaptiveVertexReconstructor") << "all tracks (" << n_tracks
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
    cout << "[AdaptiveVertexReconstructor] exception: " << e.what() << endl;
  } catch ( ... ) {
    cout << "[AdaptiveVertexReconstructor] exception" << endl;
  };
  return cleanUpVertices ( ret );
}

vector<TransientVertex> AdaptiveVertexReconstructor::cleanUpVertices (
    const vector < TransientVertex > & old ) const
{
  vector < TransientVertex > ret;
  for ( vector< TransientVertex >::const_iterator i=old.begin(); i!=old.end() ; ++i )
  {
    if (!(i->hasTrackWeight()))
    { // if we dont have track weights, we take the vtx
      ret.push_back ( *i );
      continue;
    }
    int nsig=0;
    TransientVertex::TransientTrackToFloatMap wm = i->weightMap();
    for ( TransientVertex::TransientTrackToFloatMap::const_iterator w=wm.begin(); w!=wm.end() ; ++w )
    {
      if (w->second > theWeightThreshold) nsig++;
    }
    if ( nsig > 1 ) ret.push_back ( *i );
  }
  return ret;
}
