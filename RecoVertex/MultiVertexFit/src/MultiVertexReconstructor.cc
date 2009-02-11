#include "RecoVertex/MultiVertexFit/interface/MultiVertexReconstructor.h"

using namespace std;

namespace {
  typedef MultiVertexFitter::TrackAndWeight TrackAndWeight;

  int verbose()
  {
    return 0;
  }

  void remove ( vector < TransientVertex > & vtces,
                const vector < reco::TransientTrack > & trks )
  {
    cout << "[MultiVertexReconstructor] fixme remove not yet implemented" << endl;
    // remove trks from vtces
  }

  vector < vector < TrackAndWeight > > recover (
      const vector < TransientVertex > & vtces, const vector < reco::TransientTrack > & trks )
  {
    set < reco::TransientTrack > st;
    for ( vector< reco::TransientTrack >::const_iterator i=trks.begin(); 
          i!=trks.end() ; ++i )
    {
      st.insert ( *i );
    }
    
    vector < vector < TrackAndWeight > > bundles;
    for ( vector< TransientVertex >::const_iterator vtx=vtces.begin();
          vtx!=vtces.end() ; ++vtx )
    {
      vector < reco::TransientTrack > trks = vtx->originalTracks();
      vector < TrackAndWeight > tnws;
      for ( vector< reco::TransientTrack >::const_iterator trk=trks.begin(); 
            trk!=trks.end() ; ++trk )
      {
        float w = vtx->trackWeight ( *trk ); 
        if ( w > 1e-5 )
        {
          TrackAndWeight tmp ( *trk, w );
          set < reco::TransientTrack >::iterator pos = st.find( *trk );
          if ( pos != st.end() )
          {
            st.erase ( pos  );
          }
          tnws.push_back ( tmp );
        };
      };
      bundles.push_back ( tnws );
    };

    if ( bundles.size() == 0 ) return bundles;

    // now add not-yet assigned tracks
    for ( set < reco::TransientTrack >::const_iterator i=st.begin(); 
          i!=st.end() ; ++i )
    {
      // cout << "[MultiVertexReconstructor] recovering " << i->id() << endl;
      TrackAndWeight tmp ( *i, 0. );
      bundles[0].push_back ( tmp );
    }
    return bundles;
  }
}

MultiVertexReconstructor::MultiVertexReconstructor ( 
    const VertexReconstructor & o, const AnnealingSchedule & s, float revive  ) : 
    theOldReconstructor ( o.clone() ), theFitter ( 
        MultiVertexFitter ( s, DefaultLinearizationPointFinder(), revive ) )
{
}

MultiVertexReconstructor::~MultiVertexReconstructor()
{
  delete theOldReconstructor;
}

MultiVertexReconstructor * MultiVertexReconstructor::clone() const
{
  return new MultiVertexReconstructor ( * this );
}

MultiVertexReconstructor::MultiVertexReconstructor ( 
    const MultiVertexReconstructor & o ) :
  theOldReconstructor ( o.theOldReconstructor->clone() ),
  theFitter ( o.theFitter )
{}

vector < TransientVertex > MultiVertexReconstructor::vertices ( 
    const vector < reco::TransientTrack > & trks,
    const reco::BeamSpot & s ) const
{
  return vertices ( trks );
}

vector < TransientVertex > MultiVertexReconstructor::vertices ( 
    const vector < reco::TransientTrack > & trks ) const
{
  /*
  cout << "[MultiVertexReconstructor] input trks: ";
  for ( vector< reco::TransientTrack >::const_iterator i=trks.begin(); 
        i!=trks.end() ; ++i )
  {
    cout << i->id() << "  ";
  }
  cout << endl;*/
  vector < TransientVertex > tmp = theOldReconstructor->vertices ( trks );
  if ( verbose() )
  {
    cout << "[MultiVertexReconstructor] non-freezing seeder found " << tmp.size()
         << " vertices from " << trks.size() << " tracks." << endl;
  }
  vector < vector < TrackAndWeight > > rc = recover ( tmp, trks );
  vector < CachingVertex<5> > cvts = theFitter.vertices ( rc );
  vector < TransientVertex > ret;
  for ( vector< CachingVertex<5> >::const_iterator i=cvts.begin(); 
        i!=cvts.end() ; ++i )
  {
    ret.push_back ( *i );
  };

  if ( verbose() )
  {
    cout << "[MultiVertexReconstructor] input " << tmp.size()
         << " vertices, output " << ret.size() << " vertices."
         << endl;
  };
  return ret;
}

vector < TransientVertex > MultiVertexReconstructor::vertices ( 
    const vector < reco::TransientTrack > & trks,
    const vector < reco::TransientTrack > & primaries ) const
{
  /*
  cout << "[MultiVertexReconstructor] with " << primaries.size()
       << " primaries!" << endl;
       */
  
  map < reco::TransientTrack, bool > st;
  
  vector < reco::TransientTrack > total = trks;
  for ( vector< reco::TransientTrack >::const_iterator i=trks.begin(); 
        i!=trks.end() ; ++i )
  {
    st[(*i)]=true;
  }

  // cout << "[MultiVertexReconstructor] FIXME dont just add up tracks. superpose" << endl;
  for ( vector< reco::TransientTrack >::const_iterator i=primaries.begin(); 
        i!=primaries.end() ; ++i )
  {
    if (!(st[(*i)]))
    {
      total.push_back ( *i );
    }
  }

  vector < TransientVertex > tmp = theOldReconstructor->vertices ( total );

  if ( verbose() )
  {
    cout << "[MultiVertexReconstructor] freezing seeder found " << tmp.size()
         << " vertices from " << total.size() << " tracks." << endl;
  }
  vector < vector < TrackAndWeight > > rc = recover ( tmp, trks );
  vector < CachingVertex<5> > cvts = theFitter.vertices ( rc, primaries );
   
  vector < TransientVertex > ret;
  for ( vector< CachingVertex<5> >::const_iterator i=cvts.begin(); 
        i!=cvts.end() ; ++i )
  {
    ret.push_back ( *i );
  };
  if ( verbose() )
  {
    cout << "[MultiVertexReconstructor] input " << tmp.size()
         << " vertices, output " << ret.size() << " vertices."
         << endl;
  };
  return ret;
}

VertexReconstructor * MultiVertexReconstructor::reconstructor() const
{
  return theOldReconstructor;
}
