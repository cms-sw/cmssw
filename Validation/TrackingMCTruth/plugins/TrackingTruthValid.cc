#include "Validation/TrackingMCTruth/interface/TrackingTruthValid.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include <cmath>

typedef edm::RefVector< std::vector<TrackingParticle> > TrackingParticleContainer;

typedef TrackingParticleRefVector::iterator               tp_iterator;
typedef TrackingParticle::g4t_iterator                   g4t_iterator;
typedef TrackingParticle::genp_iterator                 genp_iterator;
typedef TrackingVertex::genv_iterator                   genv_iterator;
typedef TrackingVertex::g4v_iterator                     g4v_iterator;


void TrackingTruthValid::beginJob(const edm::ParameterSet& conf) {}

TrackingTruthValid::TrackingTruthValid(const edm::ParameterSet& conf)
  : outputFile( conf.getParameter<std::string>( "outputFile" ) )
  , dbe_( NULL )
  , vec_TrackingParticle_Token_( consumes<TrackingParticleCollection>( conf.getParameter<edm::InputTag>( "src" ) ) ) {}

void TrackingTruthValid::beginRun( const edm::Run&, const edm::EventSetup& ) {
  dbe_  = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder("Tracking/TrackingMCTruth/TrackingParticle");
  

  meTPMass = dbe_->book1D("TPMass","Tracking Particle Mass",100, -1,+5.);
  meTPCharge = dbe_->book1D("TPCharge","Tracking Particle Charge",10, -5, 5);
  meTPId = dbe_->book1D("TPId","Tracking Particle Id",500, -5000, 5000);
  meTPProc = dbe_->book1D("TPProc","Tracking Particle Proc",20, -0.5, 19.5);
  meTPAllHits = dbe_->book1D("TPAllHits", "Tracking Particle All Hits", 200, -0.5, 199.5);
  meTPMatchedHits = dbe_->book1D("TPMatchedHits", "Tracking Particle Matched Hits", 100, -0.5, 99.5);
  meTPPt = dbe_->book1D("TPPt", "Tracking Particle Pt",100, 0, 100.);
  meTPEta = dbe_->book1D("TPEta", "Tracking Particle Eta",100, -7., 7.);
  meTPPhi = dbe_->book1D("TPPhi", "Tracking Particle Phi",100, -4., 4);
  meTPVtxX = dbe_->book1D("TPVtxX", "Tracking Particle VtxX",100, -100, 100.);
  meTPVtxY = dbe_->book1D("TPVtxY", "Tracking Particle VtxY",100, -100, 100.);
  meTPVtxZ = dbe_->book1D("TPVtxZ", "Tracking Particle VtxZ",100, -100, 100.);
  meTPtip = dbe_->book1D("TPtip", "Tracking Particle tip",100, 0, 1000.);
  meTPlip = dbe_->book1D("TPlip", "Tracking Particle lip",100, 0, 100.);
  
  
  // Prepare Axes Labels for Processes
  meTPProc->setBinLabel( 1,"Undefined");            // value =  0
  meTPProc->setBinLabel( 2,"Unknown");              // value =  1
  meTPProc->setBinLabel( 3,"Primary");              // value =  2
  meTPProc->setBinLabel( 4,"Hadronic");             // value =  3   
  meTPProc->setBinLabel( 5,"Decay");                // value =  4 
  meTPProc->setBinLabel( 6,"Compton");              // value =  5
  meTPProc->setBinLabel( 7,"Annihilation");         // value =  6
  meTPProc->setBinLabel( 8,"EIoni");                // value =  7
  meTPProc->setBinLabel( 9,"HIoni");                // value =  8
  meTPProc->setBinLabel(10,"MuIoni");               // value =  9
  meTPProc->setBinLabel(11,"Photon");               // value = 10
  meTPProc->setBinLabel(12,"MuPairProd");           // value = 11
  meTPProc->setBinLabel(13,"Conversions");          // value = 12
  meTPProc->setBinLabel(14,"EBrem");                // value = 13
  meTPProc->setBinLabel(15,"SynchrotronRadiation"); // value = 14
  meTPProc->setBinLabel(16,"MuBrem");               // value = 15
  meTPProc->setBinLabel(17,"MuNucl");               // value = 16
  meTPProc->setBinLabel(18,"");
  meTPProc->setBinLabel(19,"");
  meTPProc->setBinLabel(20,"");
}

void TrackingTruthValid::analyze(const edm::Event& event, const edm::EventSetup& c){

  edm::Handle<TrackingParticleCollection>  TruthTrackContainer ;
  //  edm::Handle<TrackingVertexCollection>    TruthVertexContainer;

  event.getByToken( vec_TrackingParticle_Token_, TruthTrackContainer );
  
  const TrackingParticleCollection *tPC   = TruthTrackContainer.product();

  // Loop over TrackingParticle's
  for (TrackingParticleCollection::const_iterator t = tPC -> begin(); t != tPC -> end(); ++t) {
    //if(t -> trackerPSimHit().size() ==0) cout << " Track with 0 SimHit " << endl;

    meTPMass->Fill(t->mass());
    meTPCharge->Fill(t->charge() );
    meTPId->Fill(t->pdgId());
    meTPPt->Fill(sqrt(t->momentum().perp2()));
    meTPEta->Fill(t->momentum().eta());
    meTPPhi->Fill(t->momentum().Phi());
    //#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
    //#ifdef REMOVED_JUST_TO_GET_IT_TO_COMPILE__THIS_CODE_NEEDS_TO_BE_CHECKED
    //    std::vector<PSimHit> trackerPSimHit( t->trackPSimHit(DetId::Tracker) );
    //#endif
    meTPAllHits->Fill(t->numberOfTrackerHits());
    //get the process of the first hit
    //#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
    //#ifdef REMOVED_JUST_TO_GET_IT_TO_COMPILE__THIS_CODE_NEEDS_TO_BE_CHECKED
    //    if(trackerPSimHit.size() !=0) meTPProc->Fill( trackerPSimHit.front().processType());
    //#endif
    
    // there is no more the PSimHits collection !!! how to deal w/ the processType ?
    //    if(t->numberOfTrackerHits() !=0) meTPProc->Fill( trackerPSimHit.front().processType());

    meTPMatchedHits->Fill(t->numberOfTrackerLayers());
    meTPVtxX->Fill(t->vx());
    meTPVtxY->Fill(t->vy());
    meTPVtxZ->Fill(t->vz());
    meTPtip->Fill(sqrt(t->vertex().perp2()));
    meTPlip->Fill(t->vz());

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
    cout << " Hits for this track: " << t -> trackerPSimHit().size() << endl;
    */

    /*
      std::cout << std::endl << "### Tracking Particle ###" << std::endl;
      std::cout << (*t) << std::endl;
      std::cout << "\t Tracker: " << t->trackerPSimHit().size() << std::endl;
      std::cout << "\t Muon: "    << t->muonPSimHit().size()    << std::endl;
      std::cout << (*t) << std::endl;
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
