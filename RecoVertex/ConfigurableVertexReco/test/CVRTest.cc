// user includes
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

// system includes
#include <vector>
#include <iostream>

class CVRTest : public edm::one::EDAnalyzer<> {
  /**
   *  Class that glues the combined btagging algorithm to the framework
   */
public:
  explicit CVRTest(const edm::ParameterSet&);
  ~CVRTest();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  void discussPrimary(const edm::Event&) const;

private:
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> estoken_ttk;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> estoken_mf;

  ConfigurableVertexReconstructor* vrec_;
  std::string trackcoll_;
  std::string vertexcoll_;
  std::string beamspot_;
};

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
    if (vtx.originalTracks().size() && vtx.hasRefittedTracks()) {
      cout << "    `- 1st trk: ";
      reco::TransientTrack t = vtx.originalTracks()[0];
      TrajectoryStateOnSurface tsos = t.impactPointState();
      printTSOS(tsos);
      if (vtx.refittedTracks().size()) {
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
    : estoken_ttk(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      estoken_mf(esConsumes()),
      trackcoll_(iconfig.getParameter<string>("trackcoll")),
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
  if (retColl->size()) {
    const reco::Vertex& vtx = *(retColl->begin());
    cout << "[CVRTest] persistent primary: " << vtx.x() << ", " << vtx.y() << ", " << vtx.z() << endl;
  }
}

void CVRTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::EventNumber_t const evt = iEvent.id().event();
  cout << "[CVRTest] next event: " << evt << endl;
  edm::ESHandle<MagneticField> magneticField = iSetup.getHandle(estoken_mf);
  edm::ESHandle<TransientTrackBuilder> builder = iSetup.getHandle(estoken_ttk);

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
