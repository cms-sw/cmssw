#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "SimTracker/VertexAssociation/test/testVertexAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/Records/interface/VertexAssociatorRecord.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimTracker/VertexAssociation/interface/VertexAssociatorBase.h"

#include <memory>
#include <iostream>
#include <string>

//class TrackAssociator;
class TrackAssociatorByHits;
class TrackerHitAssociator;

using namespace reco;
using namespace std;
using namespace edm;


testVertexAssociator::testVertexAssociator(edm::ParameterSet const& conf) {
}

testVertexAssociator::~testVertexAssociator() {
}

void testVertexAssociator::beginJob(const EventSetup & setup) {

  edm::ESHandle<MagneticField> theMF;
  setup.get<IdealMagneticFieldRecord>().get(theMF);
  edm::ESHandle<TrackAssociatorBase> theChiAssociator;
  setup.get<TrackAssociatorRecord>().get("TrackAssociatorByChi2",theChiAssociator);
  associatorByChi2 = (TrackAssociatorBase *) theChiAssociator.product();
  edm::ESHandle<TrackAssociatorBase> theHitsAssociator;
  setup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",theHitsAssociator);
  associatorByHits = (TrackAssociatorBase *) theHitsAssociator.product();

  edm::ESHandle<VertexAssociatorBase> theTracksAssociator;
  setup.get<VertexAssociatorRecord>().get("VertexAssociatorByTracks",theTracksAssociator);
  associatorByTracks = (VertexAssociatorBase *) theTracksAssociator.product();

  rootFile = new TFile("testVertexAssociator.root","RECREATE");
  rootFile->cd();

  xMiss = new TH1F("rs_xmiss","x Miss Distance (cm)",100,-0.05,0.05);
  yMiss = new TH1F("rs_ymiss","y Miss Distance (cm)",100,-0.05,0.05);
  zMiss = new TH1F("rs_zmiss","z Miss Distance (cm)",100,-0.05,0.05);
  rMiss = new TH1F("rs_rmiss","r Miss Distance (cm)",100,-0.05,0.05);

  zVert = new TH1F("rs_zvert","z, Reconstructed Vertex (cm)", 200, -25.0,25.0);
  zTrue = new TH1F("rs_ztrue","z, Simulated Vertex (cm)",     200, -25.0,25.0);

  nTrue = new TH1F("rs_ntrue","# of tracks, Simulated",    51,-0.5,50.5);
  nReco = new TH1F("rs_nreco","# of tracks, Reconstructed",51,-0.5,50.5);

  sr_xMiss = new TH1F("sr_xmiss","x Miss Distance (cm)",100,-0.05,0.05);
  sr_yMiss = new TH1F("sr_ymiss","y Miss Distance (cm)",100,-0.05,0.05);
  sr_zMiss = new TH1F("sr_zmiss","z Miss Distance (cm)",100,-0.05,0.05);
  sr_rMiss = new TH1F("sr_rmiss","r Miss Distance (cm)",100,-0.05,0.05);

  sr_zVert = new TH1F("sr_zvert","z, Reconstructed Vertex (cm)", 200, -25.0,25.0);
  sr_zTrue = new TH1F("sr_ztrue","z, Simulated Vertex (cm)",     200, -25.0,25.0);

  sr_nTrue = new TH1F("sr_ntrue","# of tracks, Simulated",    101,-0.5,100.5);
  sr_nReco = new TH1F("sr_nreco","# of tracks, Reconstructed",101,-0.5,100.5);

  rs_qual = new TH1F("rs_qual","Quality of Match",51,-0.01,1.01);
  sr_qual = new TH1F("sr_qual","Quality of Match",51,-0.01,1.01);

}

void testVertexAssociator::endJob() {
  cout << "Writing histos" << endl;
//  rootFile->cd();
//  xMiss->Write();
  rootFile->Write();
  rootFile->Close();
}

void testVertexAssociator::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
  using namespace edm;
  using namespace reco;

  Handle<reco::TrackCollection> trackCollectionH;
  event.getByLabel("ctfWithMaterialTracks",trackCollectionH);
  const  reco::TrackCollection  tC = *(trackCollectionH.product());

  Handle<SimTrackContainer> simTrackCollection;
  event.getByLabel("g4SimHits", simTrackCollection);
  const SimTrackContainer simTC = *(simTrackCollection.product());

  Handle<SimVertexContainer> simVertexCollection;
  event.getByLabel("g4SimHits", simVertexCollection);
  const SimVertexContainer simVC = *(simVertexCollection.product());

  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  event.getByLabel("trackingtruth","TrackTruth",TPCollectionH);
  const TrackingParticleCollection tPC   = *(TPCollectionH.product());

  edm::Handle<TrackingVertexCollection>  TVCollectionH ;
  event.getByLabel("trackingtruth","VertexTruth",TVCollectionH);
  const TrackingVertexCollection tVC   = *(TVCollectionH.product());

  edm::Handle<reco::VertexCollection>  primaryVertexH ;
  event.getByLabel("offlinePrimaryVerticesFromCTFTracks","",primaryVertexH);
  const reco::VertexCollection primaryVertexCollection   = *(primaryVertexH.product());

  //RECOTOSIM
  cout << "                      ****************** Reco To Sim ****************** " << endl;
  cout << "-- Associator by hits --" << endl;
  reco::RecoToSimCollection p = associatorByHits->associateRecoToSim (trackCollectionH,TPCollectionH,&event );

  reco::SimToRecoCollection s2rTracks = associatorByHits->associateSimToReco (trackCollectionH,TPCollectionH,&event );
//    associatorByChi2->associateRecoToSim (trackCollectionH,TPCollectionH,&event );
  cout << " Running Reco To Sim" << endl;
  reco::VertexRecoToSimCollection vR2S = associatorByTracks ->
      associateRecoToSim(primaryVertexH,TVCollectionH,event,p);
  cout << " Running Sim To Reco" << endl;
  reco::VertexSimToRecoCollection vS2R = associatorByTracks ->
      associateSimToReco(primaryVertexH,TVCollectionH,event,s2rTracks);

  cout << " Analyzing Reco To Sim" << endl;

  for (reco::VertexRecoToSimCollection::const_iterator iR2S = vR2S.begin();
       iR2S != vR2S.end(); ++iR2S) {
    math::XYZPoint recoPos = (iR2S -> key) -> position();
    double nreco = (iR2S -> key)->tracksSize();
    std::vector<std::pair<TrackingVertexRef, double> > vVR = iR2S -> val;
    for (std::vector<std::pair<TrackingVertexRef, double> >::const_iterator
        iMatch = vVR.begin(); iMatch != vVR.end(); ++iMatch) {
        TrackingVertexRef trueV =  iMatch->first;
        math::XYZTLorentzVectorD simVec = (iMatch->first)->position();
        double ntrue = trueV->daughterTracks().size();
        math::XYZPoint simPos = math::XYZPoint(simVec.x(),simVec.y(),simVec.z());
        double qual  = iMatch->second;
        double xmiss = simPos.X() - recoPos.X();
        double ymiss = simPos.Y() - recoPos.Y();
        double zmiss = simPos.Z() - recoPos.Z();
        double rmiss = sqrt(xmiss*xmiss+ymiss*ymiss+zmiss*zmiss);

        xMiss->Fill(simPos.X() - recoPos.X());
        yMiss->Fill(simPos.Y() - recoPos.Y());
        zMiss->Fill(simPos.Z() - recoPos.Z());
        rMiss->Fill(rmiss);

        zVert->Fill(simPos.Z());
        zTrue->Fill(recoPos.Z());

        nTrue->Fill(ntrue);
        nReco->Fill(nreco);
        rs_qual->Fill(qual);
    }
  }

  cout << " Analyzing Sim To Reco" << endl;

  for (reco::VertexSimToRecoCollection::const_iterator iS2R = vS2R.begin();
       iS2R != vS2R.end(); ++iS2R) {

    TrackingVertexRef simVertex = (iS2R -> key);
    math::XYZTLorentzVectorD simVec = simVertex->position();
    math::XYZPoint   simPos = math::XYZPoint(simVec.x(),simVec.y(),simVec.z());
        double ntrue = simVertex->daughterTracks().size();
//    double ntrue = simVertex->nDaughterTracks();
    std::vector<std::pair<VertexRef, double> > recoVertices = iS2R->val;
    for (std::vector<std::pair<VertexRef, double> >::const_iterator iMatch = recoVertices.begin();
         iMatch != recoVertices.end(); ++iMatch) {
      VertexRef recoV = iMatch->first;
      double qual  = iMatch->second;
      math::XYZPoint recoPos = (iMatch -> first) -> position();
      double nreco = (iMatch->first)->tracksSize();

      double xmiss = simPos.X() - recoPos.X();
      double ymiss = simPos.Y() - recoPos.Y();
      double zmiss = simPos.Z() - recoPos.Z();
      double rmiss = sqrt(xmiss*xmiss+ymiss*ymiss+zmiss*zmiss);

      sr_xMiss->Fill(simPos.X() - recoPos.X());
      sr_yMiss->Fill(simPos.Y() - recoPos.Y());
      sr_zMiss->Fill(simPos.Z() - recoPos.Z());
      sr_rMiss->Fill(rmiss);

      sr_zVert->Fill(simPos.Z());
      sr_zTrue->Fill(recoPos.Z());

      sr_nTrue->Fill(ntrue);
      sr_nReco->Fill(nreco);
      sr_qual->Fill(qual);
    }
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testVertexAssociator);



