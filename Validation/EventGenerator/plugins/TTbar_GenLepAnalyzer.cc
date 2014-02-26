#include "Validation/EventGenerator/interface/TTbar_GenLepAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "Validation/EventGenerator/interface/PdtPdgMini.h"

TTbar_GenLepAnalyzer::TTbar_GenLepAnalyzer(const edm::ParameterSet& iConfig):
  leps_(iConfig.getParameter<edm::InputTag>("leptons"))
{
   //now do what ever initialization is needed
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();

  lepsToken_=consumes< edm::View<reco::Candidate> >(leps_);

}


TTbar_GenLepAnalyzer::~TTbar_GenLepAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
TTbar_GenLepAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Handle to the Leptons collections
  edm::Handle< edm::View<reco::Candidate> > leps;
  iEvent.getByToken(lepsToken_, leps);
  if(!leps.isValid()) return;

  // loop Jet collection and fill histograms
  int nleps = 0;
  for(edm::View<reco::Candidate>::const_iterator lep_it=leps->begin(); lep_it!=leps->end(); ++lep_it){

    ++nleps;

    if (nleps > 0) { hists_["lepPtAll" ]->Fill( lep_it->p4().pt()  );
      hists_["lepEtaAll"]->Fill( lep_it->p4().eta() );
    }
    if (nleps == 1) { hists_["lepPt1" ]->Fill( lep_it->p4().pt()  );
      hists_["lepEta1"]->Fill( lep_it->p4().eta() );
    }
    if (nleps == 2) { hists_["lepPt2" ]->Fill( lep_it->p4().pt()  );
      hists_["lepEta2"]->Fill( lep_it->p4().eta() );
    }
    if (nleps == 3) { hists_["lepPt3" ]->Fill( lep_it->p4().pt()  );
      hists_["lepEta3"]->Fill( lep_it->p4().eta() );
    }
    if (nleps == 4) { hists_["lepPt4" ]->Fill( lep_it->p4().pt()  );
      hists_["lepEta4"]->Fill( lep_it->p4().eta() );
    }
  }

  hists_["lepN"     ]->Fill( nleps ) ;


}


// ------------ method called once each job just before starting event loop  ------------
void 
TTbar_GenLepAnalyzer::beginJob()
{
  if(!dbe) return;
  dbe->setCurrentFolder("Generator/TTbar");
  hists_["lepN"     ] = dbe->book1D("TTbar_lepN"     , "N"  ,   10, -.5,  9.5 );

  hists_["lepPtAll" ] = dbe->book1D("TTbar_lepPtAll_"+leps_.label() , "pt" , 1000,  0., 1000.);
  hists_["lepPt1"   ] = dbe->book1D("TTbar_lepPt1_"+leps_.label()   , "pt" , 1000,  0., 1000.);
  hists_["lepPt2"   ] = dbe->book1D("TTbar_lepPt2_"+leps_.label()   , "pt" , 1000,  0., 1000.);
  hists_["lepPt3"   ] = dbe->book1D("TTbar_lepPt3_"+leps_.label()   , "pt" , 1000,  0., 1000.);
  hists_["lepPt4"   ] = dbe->book1D("TTbar_lepPt4_"+leps_.label()   , "pt" , 1000,  0., 1000.);

  hists_["lepEtaAll"] = dbe->book1D("TTbar_lepEtaAll"+leps_.label(), "eta",  100, -5.,    5.);
  hists_["lepEta1"  ] = dbe->book1D("TTbar_lepEta1"+leps_.label()  , "eta",  100, -5.,    5.);
  hists_["lepEta2"  ] = dbe->book1D("TTbar_lepEta2"+leps_.label()  , "eta",  100, -5.,    5.);
  hists_["lepEta3"  ] = dbe->book1D("TTbar_lepEta3"+leps_.label()  , "eta",  100, -5.,    5.);
  hists_["lepEta4"  ] = dbe->book1D("TTbar_lepEta4"+leps_.label()  , "eta",  100, -5.,    5.);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TTbar_GenLepAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
TTbar_GenLepAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
TTbar_GenLepAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
TTbar_GenLepAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
TTbar_GenLepAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TTbar_GenLepAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
