#include <vector>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "RecoVertex/ConfigurableVertexReco/test/CVRAnalysis.h"
#include <iostream>

using namespace std;
using namespace reco;
using namespace edm;


namespace {
  void printTSOS ( const TrajectoryStateOnSurface & tsos )
  {
    cout << tsos.globalPosition() << " , "
         << tsos.globalMomentum() << endl;
  }

  void printVertex ( const TransientVertex & vtx )
  {
    cout << " `- pos=(" << vtx.position().x() << ", "
         << vtx.position().y() << ", " << vtx.position().z()
         << ") chi2=" << vtx.totalChiSquared()
         << " ndf=" << vtx.degreesOfFreedom() << " hr="
         << vtx.hasRefittedTracks() << endl;
    if ( vtx.originalTracks().size() && vtx.hasRefittedTracks() )
    {
      cout << "    `- 1st trk: ";
      reco::TransientTrack t = vtx.originalTracks()[0];
      TrajectoryStateOnSurface tsos = t.impactPointState();
      printTSOS ( tsos );
      if ( vtx.refittedTracks().size() )
      {
        cout << "     `- 1st refttd: ";
        reco::TransientTrack t2 = vtx.refittedTracks()[0];
        printTSOS ( t2.impactPointState() );
      }
    }
  }

  void printVertices ( const vector < TransientVertex > & vtces )
  {
    cout << "[CVRAnalysis] " << vtces.size() << " vertices." << endl;
    for ( vector< TransientVertex >::const_iterator i=vtces.begin();
          i!=vtces.end() ; ++i )
    {
      printVertex ( *i );
      cout << endl;
    }
  }

  void discussBeamSpot ( const reco::BeamSpot & bs )
  {
    cout << "[CVRAnalysis] beamspot at " << bs.position() << endl;
    reco::BeamSpot::Covariance3DMatrix cov = bs.rotatedCovariance3D();
    cout << "[CVRAnalysis] cov=" <<  cov << endl;
  }
}

CVRAnalysis::CVRAnalysis(const edm::ParameterSet& iconfig) :
  trackcoll_( iconfig.getParameter<string>("trackcoll") ),
  vertexcoll_( iconfig.getParameter<string>("vertexcoll") ),
  beamspot_( iconfig.getParameter<string>("beamspot") ),
  trackingtruth_ ( iconfig.getParameter< edm::InputTag >("truth") ),
  associator_ ( iconfig.getParameter<string>("associator") ),
  histo_ ( VertexHisto ( "vertices.root", "tracks.root" ) ),
  bhisto_ ( VertexHisto ( "vertices-b.root", "tracks-b.root" ) )
{
  edm::ParameterSet vtxconfig = iconfig.getParameter<edm::ParameterSet>("vertexreco");
  vrec_ = new ConfigurableVertexReconstructor ( vtxconfig );
  cout << "[CVRAnalysis] vtxconfig=" << vtxconfig << endl;
}

CVRAnalysis::~CVRAnalysis() {
  if ( vrec_ ) delete vrec_;
}

void CVRAnalysis::discussPrimary( const edm::Event& iEvent ) const
{
  edm::Handle<reco::VertexCollection> retColl;
  iEvent.getByLabel( vertexcoll_, retColl);
  if ( retColl->size() )
  {
    const reco::Vertex & vtx = *(retColl->begin());
    cout << "[CVRAnalysis] persistent primary: " << vtx.x() << ", " << vtx.y()
         << ", " << vtx.z() << endl;
  }
}

void CVRAnalysis::analyze( const edm::Event & iEvent,
                       const edm::EventSetup & iSetup )
{
  int evt=iEvent.id().event();
  cout << "[CVRAnalysis] next event: " << evt << endl;
  edm::ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",builder );

  edm::Handle< edm::View < reco::Track > > tks;
  iEvent.getByLabel( trackcoll_, tks );
  discussPrimary( iEvent );

  edm::Handle<TrackingVertexCollection> TVCollectionH;
  iEvent.getByLabel( trackingtruth_, TVCollectionH);

  edm::Handle<TrackingParticleCollection>  TPCollectionH;
  iEvent.getByLabel( trackingtruth_, TPCollectionH);

  edm::Handle<reco::TrackToTrackingParticleAssociator> byHitsAssociator;
  iEvent.getByLabel(associator_, byHitsAssociator);

  reco::RecoToSimCollection p =
  byHitsAssociator->associateRecoToSim ( tks, TPCollectionH );

  edm::Handle<reco::BeamSpot > bs;
  iEvent.getByLabel ( beamspot_, bs );
  discussBeamSpot ( *bs );

  vector<reco::TransientTrack> ttks;
  ttks = builder->build(tks);
  cout << "[CVRAnalysis] got " << ttks.size() << " tracks." << endl;

  cout << "[CVRAnalysis] fit w/o beamspot constraint" << endl;
  vector < TransientVertex > vtces = vrec_->vertices ( ttks );
  printVertices ( vtces );

  if ( vtces.size() && TVCollectionH->size() )
  {
    histo_.analyse ( *(TVCollectionH->begin()), vtces[0], "Primaries" );
    histo_.saveTracks ( vtces[0], p, "VtxTk" );
  }

  cout << "[CVRAnalysis] fit w beamspot constraint" << endl;
  vector < TransientVertex > bvtces = vrec_->vertices ( ttks, *bs );
  printVertices ( bvtces );
  if ( bvtces.size() && TVCollectionH->size() )
  {
    bhisto_.analyse ( *(TVCollectionH->begin()), bvtces[0], "Primaries" );
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CVRAnalysis);
