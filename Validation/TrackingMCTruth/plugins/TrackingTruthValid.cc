#include "Validation/TrackingMCTruth/interface/TrackingTruthValid.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

using namespace std;
using namespace ROOT::Math;
using namespace edm;

typedef edm::RefVector< std::vector<TrackingParticle> > TrackingParticleContainer;
typedef std::vector<TrackingParticle>                   TrackingParticleCollection;

typedef TrackingParticleRefVector::iterator               tp_iterator;
typedef TrackingParticle::g4t_iterator                   g4t_iterator;
typedef TrackingParticle::genp_iterator                 genp_iterator;
typedef TrackingVertex::genv_iterator                   genv_iterator;
typedef TrackingVertex::g4v_iterator                     g4v_iterator;


void TrackingTruthValid::beginJob(const edm::ParameterSet& conf) {}

TrackingTruthValid::TrackingTruthValid(const edm::ParameterSet& conf) {
  
  outputFile = conf.getUntrackedParameter<string>("outputFile","trackingtruthhisto.root");
  src_ =  conf.getParameter<edm::InputTag>( "src" );
  
  dbe_  = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe_->showDirStructure();
  dbe_->setCurrentFolder("TrackingMCTruth/TrackingParticle");
  

  meTPMass = dbe_->book1D("TPMass","Tracking Particle Mass",100, -1,+5.);
  meTPCharge = dbe_->book1D("TPCharge","Tracking Particle Charge",10, -5, 5);
  meTPId = dbe_->book1D("TPId","Tracking Particle Id",500, -5000, 5000);
  meTPAllHits = dbe_->book1D("TPAllHits", "Tracking Particle All Hits", 200, 0, 200);
  meTPMatchedHits = dbe_->book1D("TPMatchedHits", "Tracking Particle Matched Hits", 100, 0, 100);
  meTPPt = dbe_->book1D("TPPt", "Tracking Particle Pt",100, 0, 100.);
  meTPEta = dbe_->book1D("TPEta", "Tracking Particle Eta",100, -7., 7.);
  meTPPhi = dbe_->book1D("TPPhi", "Tracking Particle Phi",100, -4., 4);
  meTPVtxX = dbe_->book1D("TPVtxX", "Tracking Particle VtxX",100, -100, 100.);
  meTPVtxY = dbe_->book1D("TPVtxY", "Tracking Particle VtxY",100, -100, 100.);
  meTPVtxZ = dbe_->book1D("TPVtxZ", "Tracking Particle VtxZ",100, -100, 100.);
  meTPtip = dbe_->book1D("TPtip", "Tracking Particle tip",100, 0, 1000.);
  meTPlip = dbe_->book1D("TPlip", "Tracking Particle lip",100, 0, 100.);
}

void TrackingTruthValid::analyze(const edm::Event& event, const edm::EventSetup& c){
  using namespace std;

  edm::Handle<TrackingParticleCollection>  TruthTrackContainer ;
  edm::Handle<TrackingVertexCollection>    TruthVertexContainer;
  event.getByLabel(src_,TruthTrackContainer );
  event.getByType(TruthVertexContainer);

  const TrackingParticleCollection *tPC   = TruthTrackContainer.product();
  const TrackingVertexCollection   *tVC   = TruthVertexContainer.product();

  /*
  // Get and print HepMC event for comparison
  edm::Handle<edm::HepMCProduct> hepMC;
  event.getByLabel("source",hepMC);
  const edm::HepMCProduct *mcp = hepMC.product();
  //  const HepMC::GenEvent *genEvent = mcp -> GetEvent();
  */

  cout << "Found " << tPC -> size() << " tracks and " << tVC -> size() << " vertices." <<endl;

// Loop over TrackingParticle's

  for (TrackingParticleCollection::const_iterator t = tPC -> begin(); t != tPC -> end(); ++t) {


    meTPMass->Fill(t->mass());
    meTPCharge->Fill(t->charge() );
    meTPId->Fill(t->pdgId());
    meTPPt->Fill(sqrt(t->momentum().perp2()));
    meTPEta->Fill(t->momentum().eta());
    meTPPhi->Fill(t->momentum().Phi());
    meTPAllHits->Fill(t->trackPSimHit().size());
    meTPMatchedHits->Fill(t->matchedHit());
    meTPVtxX->Fill(sqrt(t->vertex().x()));
    meTPVtxY->Fill(sqrt(t->vertex().y()));
    meTPVtxZ->Fill(sqrt(t->vertex().z()));
    meTPtip->Fill(sqrt(t->vertex().perp2()));
    meTPlip->Fill(sqrt(t->vertex().z()));

    /*
   // Compare momenta from sources
    cout << "T.P.   Track mass, Momentum, q , ID, & Event # "
          << t -> mass()  << " " 
          << t -> p4()    << " " << t -> charge() << " "
          << t -> pdgId() << " "
          << t -> eventId().bunchCrossing() << "." << t -> eventId().event() << endl;

    if(t->mass() < 0) cout << "======= WARNING, this particle has negative mass: " << t->mass()  
			   << " and pdgId: " << t->pdgId() << endl;
    if(t->pdgId() == 0) cout << "======= WARNING, this particle has pdgId = 0: "     << t->pdgId() << endl;
    cout << " Hits for this track: " << t -> trackPSimHit().size() << endl;
    */


  }  // End loop over TrackingParticle
  
  // Loop over TrackingVertex's
  /*  
  cout << "Dumping sample vertex info" << endl;
  for (TrackingVertexCollection::const_iterator v = tVC -> begin(); v != tVC -> end(); ++v) {
    cout << " Vertex Position & Event #" << v -> position() << " " << v -> eventId().bunchCrossing() << "." << v -> eventId().event() << endl;
    cout << "  Associated with " << v -> daughterTracks().size() << " tracks" << endl;
    // Get Geant and HepMC positions
    for (genv_iterator genV = v -> genVertices_begin(); genV != v -> genVertices_end(); ++genV) {
      cout << "  HepMC vertex position " << (*(*genV)).position() << endl;
    }
    for (g4v_iterator g4V = v -> g4Vertices_begin(); g4V != v -> g4Vertices_end(); ++g4V) {
      cout << "  Geant vertex position " << (*g4V).position() << endl;
      // Probably empty all the time, currently
    }

    // Loop over daughter track(s)
    for (tp_iterator iTP = v -> daughterTracks_begin(); iTP != v -> daughterTracks_end(); ++iTP) {
      cout << "  Daughter starts:      " << (*(*iTP)).vertex();
      for (g4t_iterator g4T  = (*(*iTP)).g4Track_begin(); g4T != (*(*iTP)).g4Track_end(); ++g4T) {
        cout << " p " << g4T->momentum();
      }
      cout << endl;
    }

    // Loop over source track(s) (can be multiple since vertices are collapsed)
    for (tp_iterator iTP = v -> sourceTracks_begin(); iTP != v -> sourceTracks_end(); ++iTP) {
      cout << "  Source   starts: " << (*(*iTP)).vertex();
      for (g4t_iterator g4T  = (*iTP)->g4Track_begin(); g4T != (*iTP)->g4Track_end(); ++g4T) {
        cout << ", p " <<  g4T ->momentum();
      }
      cout << endl;
    }
  }  // End loop over TrackingVertex
  */


}

void TrackingTruthValid::endJob(){ 

  if ( outputFile.size() != 0 && dbe_ ) dbe_->save(outputFile);

} 
