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

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

#include <iostream>

using namespace std;
using namespace reco;
using namespace edm;

namespace {
  void printTSOS(const TrajectoryStateOnSurface& tsos) {
    cout << tsos.globalPosition() << " , " << tsos.globalMomentum() << endl;
  }

  void printVertex(const TransientVertex& vtx) {
    cout << " `- pos=(" << vtx.position().x() << ", " << vtx.position().y() << ", " << vtx.position().z()
         << ") chi2=" << vtx.totalChiSquared() << " ndf=" << vtx.degreesOfFreedom() << " hr=" << vtx.hasRefittedTracks()
         << endl;
    if (!vtx.originalTracks().empty() && vtx.hasRefittedTracks()) {
      cout << "    `- 1st trk: ";
      reco::TransientTrack t = vtx.originalTracks()[0];
      TrajectoryStateOnSurface tsos = t.impactPointState();
      printTSOS(tsos);
      if (!vtx.refittedTracks().empty()) {
        cout << "     `- 1st refttd: ";
        reco::TransientTrack t2 = vtx.refittedTracks()[0];
        printTSOS(t2.impactPointState());
      }
    }
  }

  void printVertices(const vector<TransientVertex>& vtces) {
    cout << "[CVRTest] " << vtces.size() << " vertices." << endl;
    for (vector<TransientVertex>::const_iterator i = vtces.begin(); i != vtces.end(); ++i) {
      printVertex(*i);
      cout << endl;
    }
  }

  void discussBeamSpot(const reco::BeamSpot& bs) {
    cout << "[CVRTest] beamspot at " << bs.position() << endl;
    reco::BeamSpot::Covariance3DMatrix cov = bs.rotatedCovariance3D();
    cout << "[CVRTest] cov=" << cov << endl;
  }
}  // namespace

CVRTest::CVRTest(const edm::ParameterSet& iconfig)
    : trackcoll_(iconfig.getParameter<string>("trackcoll")),
      vertexcoll_(iconfig.getParameter<string>("vertexcoll")),
      beamspot_(iconfig.getParameter<string>("beamspot")) {
  edm::ParameterSet vtxconfig = iconfig.getParameter<edm::ParameterSet>("vertexreco");
  vrec_ = new ConfigurableVertexReconstructor(vtxconfig);
  cout << "[CVRTest] vtxconfig=" << vtxconfig << endl;
}

CVRTest::~CVRTest() {
  if (vrec_)
    delete vrec_;
}

void CVRTest::discussPrimary(const edm::Event& iEvent) const {
  edm::Handle<reco::VertexCollection> retColl;
  iEvent.getByLabel(vertexcoll_, retColl);
  if (!retColl->empty()) {
    const reco::Vertex& vtx = *(retColl->begin());
    cout << "[CVRTest] persistent primary: " << vtx.x() << ", " << vtx.y() << ", " << vtx.z() << endl;
  }
}

void CVRTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::EventNumber_t const evt = iEvent.id().event();
  cout << "[CVRTest] next event: " << evt << endl;
  edm::ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);

  edm::Handle<reco::TrackCollection> tks;
  iEvent.getByLabel(trackcoll_, tks);
  discussPrimary(iEvent);

  edm::Handle<reco::BeamSpot> bs;
  iEvent.getByLabel(beamspot_, bs);
  discussBeamSpot(*bs);

  vector<reco::TransientTrack> ttks;
  ttks = builder->build(tks);
  cout << "[CVRTest] got " << ttks.size() << " tracks." << endl;

  cout << "[CVRTest] fit w/o beamspot constraint" << endl;
  vector<TransientVertex> vtces = vrec_->vertices(ttks);
  printVertices(vtces);

  // cout << "[CVRTest] fit w beamspot constraint" << endl;
  // vector < TransientVertex > bvtces = vrec_->vertices ( ttks, *bs );
  // printVertices ( bvtces );
}

//define this as a plug-in
DEFINE_FWK_MODULE(CVRTest);
