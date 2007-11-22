#include <vector>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/ConfigurableVertexReco/test/CVRTest.h"
#include <iostream>

using namespace std;
using namespace reco;
using namespace edm;

CVRTest::CVRTest(const edm::ParameterSet& iconfig) :
  trackcoll_( iconfig.getParameter<string>("trackcoll") )
{
  edm::ParameterSet vtxconfig = iconfig.getParameter<edm::ParameterSet>("vertexreco");
  vrec_ = new ConfigurableVertexReconstructor ( vtxconfig );
  cout << "[CVRTest] will use " << vtxconfig.getParameter<string>("finder") << endl;
}

CVRTest::~CVRTest() {
  if ( vrec_ ) delete vrec_;
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

  vector<reco::TransientTrack> ttks;
  ttks = builder->build(tks);
  cout << "[CVRTest] got " << ttks.size() << " tracks." << endl;

  vector < TransientVertex > vtces = vrec_->vertices ( ttks );

  cout << "[CVRTest] " << vtces.size() << " vertices." << endl;

  for ( vector< TransientVertex >::const_iterator i=vtces.begin(); 
        i!=vtces.end() ; ++i )
  {
    cout << " `- " << i->position().x() << endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CVRTest);
