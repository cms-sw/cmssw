#include "RecoVertex/AdaptiveVertexFinder/interface/AdaptiveVertexReconstructor.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
  if ( old.hasPrior() )
  {
    std::cout << "[AdaptiveVertexReconstructor] WARNING prior is discarded!" << std::endl;
  }
  // TransientVertex ret ( old );
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
  TransientVertex ret ( old.vertexState(), newtrks,
                        old.totalChiSquared(), old.degreesOfFreedom() );
  ret.weightMap ( mp );
  return ret;
}

std::vector<TransientVertex> AdaptiveVertexReconstructor::vertices (
    const std::vector<reco::TransientTrack> & tracks ) const
{
  std::vector < TransientVertex > ret;
  std::set < reco::TransientTrack > remainingtrks;
  for ( vector< reco::TransientTrack >::const_iterator i=tracks.begin();
        i!=tracks.end() ; ++i )
  {
    remainingtrks.insert ( *i );
  }
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
      AdaptiveVertexFitter fitter ( ann );
      vector < reco::TransientTrack > fittrks;
      fittrks.reserve ( remainingtrks.size() );
      for ( set < reco::TransientTrack >::const_iterator i=remainingtrks.begin();
            i!=remainingtrks.end() ; ++i )
      {
        fittrks.push_back ( *i );
      }
      TransientVertex newvtx = cleanUp ( fitter.vertex ( fittrks ) );
      ret.push_back ( newvtx );
      erase ( newvtx, remainingtrks );
      if ( n_tracks == remainingtrks.size() )
      {
        cout << "[AdaptiveVertexReconstructor] warning: all tracks (" << n_tracks
             << ") would be recycled for next fit." << endl;
        cout << "                              breaking after reconstruction of "
             << ret.size() << " vertices." << endl;
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
