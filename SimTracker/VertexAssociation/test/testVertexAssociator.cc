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

  rootFile = new TFile("MyHistograms.root","RECREATE");
  rootFile->cd();

  xMiss = new TH1F("xmiss","x Miss Distance (cm)",100,-0.02,0.02);
  yMiss = new TH1F("ymiss","y Miss Distance (cm)",100,-0.02,0.02);
  zMiss = new TH1F("zmiss","z Miss Distance (cm)",100,-0.02,0.02);
  rMiss = new TH1F("rmiss","r Miss Distance (cm)",100,-0.02,0.02);

  zVert = new TH1F("zvert","z, Reconstructed Vertex (cm)", 200, -1.0,1.0);
  zTrue = new TH1F("ztrue","z, Simulated Vertex (cm)",     200, -1.0,1.0);

  nTrue = new TH1F("ntrue","# of tracks, Simulated",    51,-0.5,50.5);
  nReco = new TH1F("nreco","# of tracks, Reconstructed",51,-0.5,50.5);

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

  cout << "\nEvent ID = "<< event.id() << endl ;


  //RECOTOSIM
  cout << "                      ****************** Reco To Sim ****************** " << endl;
  cout << "-- Associator by hits --" << endl;
  reco::RecoToSimCollection p = associatorByHits->associateRecoToSim (trackCollectionH,TPCollectionH,&event );

  reco::SimToRecoCollection s2rTracks = associatorByHits->associateSimToReco (trackCollectionH,TPCollectionH,&event );
//    associatorByChi2->associateRecoToSim (trackCollectionH,TPCollectionH,&event );
  reco::VertexRecoToSimCollection vR2S = associatorByTracks ->
      associateRecoToSim(primaryVertexH,TVCollectionH,event,p);
  reco::VertexSimToRecoCollection vS2R = associatorByTracks ->
      associateSimToReco(primaryVertexH,TVCollectionH,event,s2rTracks);

  for (reco::VertexRecoToSimCollection::const_iterator iR2S = vR2S.begin();
       iR2S != vR2S.end(); ++iR2S) {
    math::XYZPoint recoPos = (iR2S -> key) -> position();
    double nreco = (iR2S -> key)->tracksSize();
//    cout << "Reco Position " << recoPos << endl;
    std::vector<std::pair<TrackingVertexRef, double> > vVR = iR2S -> val;
//    cout << "Found Recovertex with " << vVR.size() << " associated TrackingVertex" << endl;
    for (std::vector<std::pair<TrackingVertexRef, double> >::const_iterator
        iMatch = vVR.begin(); iMatch != vVR.end(); ++iMatch) {
//        cout << "Match found with quality " <<  iMatch -> second << endl;
        TrackingVertexRef trueV = iMatch->first;
        HepLorentzVector simVec = (iMatch -> first) -> position();
        double ntrue = trueV->daughterTracks().size();
        math::XYZPoint simPos = math::XYZPoint(simVec.x(),simVec.y(),simVec.z());

//        cout << "Sim  Position " << simPos << " distance " << (simPos - recoPos).R()   << endl;

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
    }
  }


//      //SIMTORECO
//  cout << "                      ****************** Sim To Reco ****************** " << endl;
//  cout << "-- Associator by hits --" << endl;
//  reco::SimToRecoCollection q =
//    associatorByHits->associateSimToReco(trackCollectionH,TPCollectionH,&event );

}



#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(testVertexAssociator);



