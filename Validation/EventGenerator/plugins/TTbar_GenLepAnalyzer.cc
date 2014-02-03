#include "Validation/EventGenerator/interface/TTbar_GenLepAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "Validation/EventGenerator/interface/PdtPdgMini.h"

TTbar_GenLepAnalyzer::TTbar_GenLepAnalyzer(const edm::ParameterSet& iConfig):
  leps_(iConfig.getParameter<edm::InputTag>("leptons"))
{

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


void TTbar_GenLepAnalyzer::bookHistograms(DQMStore::IBooker &i, edm::Run const &r, edm::EventSetup const &e){
  i.setCurrentFolder("Generator/TTbar");
  hists_["lepN"     ] = i.book1D("TTbar_lepN"     , "N"  ,   10, -.5,  9.5 );

  hists_["lepPtAll" ] = i.book1D("TTbar_lepPtAll_"+leps_.label() , "pt" , 1000,  0., 1000.);
  hists_["lepPt1"   ] = i.book1D("TTbar_lepPt1_"+leps_.label()   , "pt" , 1000,  0., 1000.);
  hists_["lepPt2"   ] = i.book1D("TTbar_lepPt2_"+leps_.label()   , "pt" , 1000,  0., 1000.);
  hists_["lepPt3"   ] = i.book1D("TTbar_lepPt3_"+leps_.label()   , "pt" , 1000,  0., 1000.);
  hists_["lepPt4"   ] = i.book1D("TTbar_lepPt4_"+leps_.label()   , "pt" , 1000,  0., 1000.);

  hists_["lepEtaAll"] = i.book1D("TTbar_lepEtaAll", "eta",  100, -5.,    5.);
  hists_["lepEta1"  ] = i.book1D("TTbar_lepEta1"  , "eta",  100, -5.,    5.);
  hists_["lepEta2"  ] = i.book1D("TTbar_lepEta2"  , "eta",  100, -5.,    5.);
  hists_["lepEta3"  ] = i.book1D("TTbar_lepEta3"  , "eta",  100, -5.,    5.);
  hists_["lepEta4"  ] = i.book1D("TTbar_lepEta4"  , "eta",  100, -5.,    5.);
}


