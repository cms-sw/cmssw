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
#include "RecoVertex/ConfigurableVertexReco/test/CVRTest.h"
#include <iostream>

using namespace std;
using namespace reco;
using namespace edm;


namespace {

  void printVertex ( const TransientVertex & vtx )
  {
    cout << " `- pos=(" << vtx.position().x() << ", "
         << vtx.position().y() << ", " << vtx.position().z()
         << ") chi2=" << vtx.totalChiSquared() << endl;
  }

  void printVertices ( const vector < TransientVertex > & vtces )
  {
    cout << "[CVRTest] " << vtces.size() << " vertices." << endl;
    for ( vector< TransientVertex >::const_iterator i=vtces.begin();
          i!=vtces.end() ; ++i )
    {
      printVertex ( *i );
    }
  }
}

CVRTest::CVRTest(const edm::ParameterSet& iconfig) :
  trackcoll_( iconfig.getParameter<string>("trackcoll") ),
  vertexcoll_( iconfig.getParameter<string>("vertexcoll") )
{
  edm::ParameterSet vtxconfig = iconfig.getParameter<edm::ParameterSet>("vertexreco");
  vrec_ = new ConfigurableVertexReconstructor ( vtxconfig );
  cout << "[CVRTest] will use " << vtxconfig.getParameter<string>("finder") << endl;
}

CVRTest::~CVRTest() {
  if ( vrec_ ) delete vrec_;
}

void CVRTest::discussPrimary( const edm::Event& iEvent ) const
{
  edm::Handle<reco::VertexCollection> retColl;
  iEvent.getByLabel( vertexcoll_, retColl);
  if ( retColl->size() )
  {
    const reco::Vertex & vtx = *(retColl->begin());
    cout << "[CVRTest] persistent primary: " << vtx.x() << ", " << vtx.y()
         << ", " << vtx.z() << endl;
  }
}

void CVRTest::analyze( const edm::Event& iEvent,
                       const edm::EventSetup& iSetup)
{
  int evt=iEvent.id().event();
  cout << "[CVRTest] next event: " << evt << endl;
  edm::ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",builder );

  edm::Handle<reco::TrackCollection> tks;
  iEvent.getByLabel( trackcoll_, tks );
  discussPrimary( iEvent );

  vector<reco::TransientTrack> ttks;
  ttks = builder->build(tks);
  cout << "[CVRTest] got " << ttks.size() << " tracks." << endl;

  vector < TransientVertex > vtces = vrec_->vertices ( ttks );

  printVertices ( vtces );

}

//define this as a plug-in
DEFINE_FWK_MODULE(CVRTest);
