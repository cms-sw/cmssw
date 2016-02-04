#include "RecoVertex/MultiVertexFit/interface/MultiVertexBSeeder.h"
// #include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
// #include "RecoVertex/VertexTools/interface/BeamSpot.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "RecoVertex/VertexTools/interface/FsmwModeFinder3d.h"
#include "CommonTools/Clustering1D/interface/FsmwClusterizer1D.h"
// #include "CommonTools/Clustering1D/interface/MultiClusterizer1D.h"
#include "CommonTools/Clustering1D/interface/OutermostClusterizer1D.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// #define MVBS_DEBUG
#ifdef MVBS_DEBUG
#include <map>
#include "RecoVertex/MultiVertexFit/interface/DebuggingHarvester.h"
#endif

using namespace std;

namespace {
  vector < reco::TransientTrack > convert ( 
      const vector < const reco::TransientTrack * > & ptrs )
  {
    vector < reco::TransientTrack > ret;
    for ( vector< const reco::TransientTrack * >::const_iterator i=ptrs.begin(); 
          i!=ptrs.end() ; ++i )
    {
      ret.push_back ( **i );
    }
    return ret;
  }

  int verbose()
  {
    return 0;
  }

  inline GlobalPoint & operator += ( GlobalPoint & a, const GlobalPoint & b )
  {
    a = GlobalPoint ( a.x() + b.x(), a.y() + b.y(), a.z() + b.z() );
    return a;
  }

  inline GlobalPoint & operator /= ( GlobalPoint & a, float b )
  {
    a = GlobalPoint ( a.x() / b, a.y() / b, a.z() / b );
    return a;
  }

  bool element ( const reco::TransientTrack & rt, const TransientVertex & rv )
  {
    const vector < reco::TransientTrack > trks = rv.originalTracks();
    for ( vector< reco::TransientTrack >::const_iterator i=trks.begin(); i!=trks.end() ; ++i )
    {
      if ( rt == ( *i ) ) return true;
    };
    return false;
  }

  GlobalPoint toPoint ( const GlobalVector & v )
  {
    return GlobalPoint ( v.x(), v.y(), v.z() );
  }

  GlobalVector toVector ( const GlobalPoint & v )
  {
    return GlobalVector ( v.x(), v.y(), v.z() );
  }


  GlobalPoint computeJetOrigin ( const vector < reco::TransientTrack > & trks )
  {
    FsmwModeFinder3d f;
    vector< ModeFinder3d::PointAndDistance> input;
    for ( vector< reco::TransientTrack >::const_iterator i=trks.begin();
          i!=trks.end() ; ++i )
    {
      input.push_back ( ModeFinder3d::PointAndDistance 
                        ( i->impactPointState().globalPosition(), 1. )  );
    }
    return f(input);
  }

  GlobalVector computeJetDirection ( const vector < reco::TransientTrack > & trks )
  {
    FsmwModeFinder3d f;
    vector< ModeFinder3d::PointAndDistance> input;
    for ( vector< reco::TransientTrack >::const_iterator i=trks.begin();
          i!=trks.end() ; ++i )
    {
      input.push_back ( ModeFinder3d::PointAndDistance ( 
            toPoint ( i->impactPointState().globalMomentum() ), 1. )  );
    }
    GlobalPoint pt ( f(input) );
    pt/=pt.mag();
    return toVector ( pt );
  }

  GlobalTrajectoryParameters computeJetTrajectory ( const vector < reco::TransientTrack > & trks )
  {
    /**
     *  construct a trajectory at the mean of
     *  the impact points of all tracks,
     *  momentum = total momentum of all tracks
     *
     */
    if ( trks.size() == 0 ) return GlobalTrajectoryParameters();

    GlobalVector mom = computeJetDirection ( trks );
    GlobalPoint pos = computeJetOrigin ( trks );

    GlobalTrajectoryParameters ret ( pos, mom, 0,
        &(trks[0].impactPointState().globalParameters().magneticField()) );
    #ifdef MVBS_DEBUG
    DebuggingHarvester("out.txt").save ( ret , "p<sub>tot</sub>" );
    #endif
    return ret;
  }

  vector < Cluster1D < reco::TransientTrack > > computeIPs (
      const vector < reco::TransientTrack > & trks )
  {
    GlobalTrajectoryParameters jet = computeJetTrajectory ( trks );
    FreeTrajectoryState axis ( jet );
    TwoTrackMinimumDistance ttmd;
    vector < Cluster1D < reco::TransientTrack > > pts;
    for ( vector< reco::TransientTrack >::const_iterator i=trks.begin(); 
          i!=trks.end() ; ++i )
    {
      bool status = ttmd.calculate( axis,*( i->impactPointState().freeState() ) );
      if (status) {
        pair < GlobalPoint, GlobalPoint > pt = ttmd.points();
        double d = ( pt.first - pt.second ).mag();
        double w = 1. / ( 0.002 + d ); // hard coded weights
        double s = ( pt.first - axis.position() ).mag();
        Measurement1D ms ( s, 1.0 );
        vector < const reco::TransientTrack * > trk;
        trk.push_back ( &(*i) );
        pts.push_back ( Cluster1D < reco::TransientTrack > ( ms, trk, w ) );
      }
    }
    /*
    #ifdef MVBS_DEBUG
    map < string, harvest::MultiType > attrs;
    attrs["point:mag"]=0.5;
    attrs["point:color"]="khaki";
    DebuggingHarvester("out.txt").save ( pts, jet, attrs, "ip" );
    #endif
    */
    return pts;
  }

  GlobalPoint computePos ( const GlobalTrajectoryParameters & jet,
      double s )
  {
    GlobalPoint ret = jet.position();
    ret += s * jet.momentum();
    return ret;
  }

  TransientVertex pseudoVertexFit ( const Cluster1D < reco::TransientTrack > & src,
      bool ascending=false, bool kalmanfit=false )
  {
    // cout << "[MultiVertexBSeeder] debug: pseudoVertexFit with " << flush;
    vector < const reco::TransientTrack * > trkptrs=src.tracks();
    vector < reco::TransientTrack > trks = convert ( trkptrs );
    // cout << trks.size() << " tracks.";
    GlobalPoint gp;
    GlobalError ge;
    if ( kalmanfit )
    {
        TransientVertex v = KalmanVertexFitter().vertex ( trks );
        gp=v.position();
    }
    TransientVertex ret = TransientVertex ( gp, ge, trks, -1. );
    TransientVertex::TransientTrackToFloatMap mp;
    float w=1.0; float r=0.5;
    if ( ascending ) { w = pow ( (float) 0.5, (int) (trks.size()-1) ); r=2.0; };
    for ( vector< reco::TransientTrack >::const_iterator i=trks.begin(); 
         i!=trks.end() ; ++i )
    {
      mp[*i]=w;
      w*=r;
    }
    ret.weightMap ( mp );
    // cout << "[MultiVertexBSeeder] debug: return pseudoVertexFit with " << endl;
    return ret;
  }

  /* TransientVertex kalmanVertexFit ( const Cluster1D < reco::TransientTrack > & src )
  {
    KalmanVertexFitter fitter;
      vector < const reco::TransientTrack * > trkptrs=src.tracks();
      vector < reco::TransientTrack > trks = convert ( trkptrs );
      return fitter.vertex ( trks );
    return TransientVertex();
  }*/
}

MultiVertexBSeeder * MultiVertexBSeeder::clone() const
{
  return new MultiVertexBSeeder ( * this );
}

MultiVertexBSeeder::MultiVertexBSeeder ( double nsigma ) :
  theNSigma ( nsigma ) {}

vector < TransientVertex > MultiVertexBSeeder::vertices (
    const vector < reco::TransientTrack > & trks,
    const reco::BeamSpot & s) const
{
  return vertices ( trks );
}

vector < TransientVertex > MultiVertexBSeeder::vertices (
    const vector < reco::TransientTrack > & trks ) const
{
  vector < Cluster1D < reco::TransientTrack > > ips = computeIPs ( trks );
  /*
  FsmwClusterizer1D < reco::TransientTrack > fsmw ( .4, theNSigma );
  MultiClusterizer1D<reco::TransientTrack> finder ( fsmw );
  */
  OutermostClusterizer1D<reco::TransientTrack> finder;

  pair < vector < Cluster1D<reco::TransientTrack> >, 
         vector < const reco::TransientTrack * > > res;
  res = finder ( ips );
  #ifdef MVBS_DEBUG
  // need to compute jet trajectory again :-(
  GlobalTrajectoryParameters jet = computeJetTrajectory ( trks );
  map < string, harvest::MultiType > attrs;
  attrs["point:mag"]=0.75;
  attrs["point:color"]="blue";
  attrs["point:name"]="mode";
  DebuggingHarvester("out.txt").save ( res.first, jet, attrs, "mode" );
  #endif

  vector < TransientVertex > ret;
  for ( vector< Cluster1D < reco::TransientTrack > >::const_iterator i=res.first.begin(); 
        i!=res.first.end() ; ++i )
  {
    ret.push_back ( pseudoVertexFit ( *i, i!=res.first.begin(), 
                    /* kalman fit*/ true ) );
  }
  return ret;
}

#ifdef MVBS_DEBUG
#undef MVBS_DEBUG
#endif
