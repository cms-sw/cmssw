#include "RecoVertex/ConfigurableVertexReco/test/VertexHisto.h"
#include "Workspace/DataHarvesting/interface/Writer.h"
#include "Workspace/DataHarvesting/interface/SystemWriter.h"
#include "Workspace/DataHarvesting/interface/Tuple.h"

using namespace std;
using namespace reco;
using namespace edm;
using namespace dataharvester;

void VertexHisto::stamp()
{
  if ( hasStamped ) return;
  SystemWriter ( filename_ ).save ();
  hasStamped=true;
}

VertexHisto::VertexHisto ( const string & filename, const string & trackname ) : filename_ ( filename ),
  /* tracks_ ( TrackHisto ( trackname ) ), */ hasStamped(false)
{
  stamp();
}

void VertexHisto::saveTracks ( const TransientVertex & trec, 
    const reco::RecoToSimCollection & p,
    const string & name ) const
{
  /*
  vector < TransientTrack > ttrks = trec.originalTracks();
  for ( vector< TransientTrack >::const_iterator i=ttrks.begin(); 
        i!=ttrks.end() ; ++i )
  {
    // reco::Track t = i->track();
    // reco::TrackRef k; // ( t );
    TrackRef k = i->trackBaseRef().castTo<TrackRef>();
    vector<pair<TrackingParticleRef, double> > coll = p[k];
    if ( coll.size() )
    {
      tracks_.analyse ( *(coll[0].first), (*i), "VtxTk" );
    } else {
      tracks_.analyse ( (*i), "UnassociatedTk" );
    }
  }

  if ( trec.hasRefittedTracks () )
  {
    vector < TransientTrack > ttrks = trec.refittedTracks();
    for ( vector< TransientTrack >::const_iterator i=ttrks.begin(); 
          i!=ttrks.end() ; ++i )
    {
      // reco::Track t = i->track();
      // reco::TrackRef k; // ( t );
      TrackRef k = trec.originalTrack ( *i ).trackBaseRef().castTo<TrackRef>();
      vector<pair<TrackingParticleRef, double> > coll = p[k];
      if ( coll.size() )
      {
        tracks_.analyse ( *(coll[0].first), (*i), "RefittedVtxTk" );
      } else {
        tracks_.analyse ( (*i), "UnassociatedRefittedTk" );
      }
    }
  }*/
}

void VertexHisto::analyse ( const TrackingVertex & sim, const TransientVertex & rec,
                            const string & name ) const
{
  Tuple t ( name );
  float simx=sim.position().x();
  float simy=sim.position().y();
  float simz=sim.position().z();
  float recx=rec.position().x();
  float recy=rec.position().y();
  float recz=rec.position().z();
  float dx=recx-simx;
  float dy=recy-simy;
  float dz=recz-simz;
  t["simx"]=simx;
  t["simy"]=simy;
  t["simz"]=simz;
  t["recx"]=recx;
  t["recy"]=recy;
  t["recz"]=recz;
  t["x"]=dx;
  t["y"]=dy;
  t["z"]=dz;
  t["dx"]=dx;
  t["dy"]=dy;
  t["dz"]=dz;
  float pullx = dx / sqrt ( rec.positionError().cxx() );
  float pully = dy / sqrt ( rec.positionError().cyy() );
  float pullz = dz / sqrt ( rec.positionError().czz() );
  t["stx"]=pullx;
  t["sty"]=pully;
  t["stz"]=pullz;
  t["time"]=0.;

  Writer::file ( filename_ ) << t;
}

VertexHisto::~VertexHisto()
{
  Writer::close();
}
