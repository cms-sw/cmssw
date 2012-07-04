#include "Validation/EventGenerator/interface/TTbar_GenJetAnalyzer.h"


TTbar_GenJetAnalyzer::TTbar_GenJetAnalyzer(const edm::ParameterSet& iConfig):
  jets_(iConfig.getParameter<edm::InputTag>("jets"))
{
   //now do what ever initialization is needed
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
}


TTbar_GenJetAnalyzer::~TTbar_GenJetAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void

TTbar_GenJetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
 
  // --- the MC weights ---
  Handle<GenEventInfoProduct> evt_info;
  iEvent.getByType(evt_info);
  weight = evt_info->weight() ;

  // Gather information in the GenJet collection
  edm::Handle<std::vector<reco::GenJet> > jets;
  iEvent.getByLabel(jets_, jets );

  // loop Jet collection and fill histograms
  int njets = 0;
  for(std::vector<reco::GenJet>::const_iterator jet_it=jets->begin(); jet_it!=jets->end(); ++jet_it){  

    ++njets; 

    hists_["jetPtAll" ]->Fill( jet_it->pt() , weight );
    hists_["jetEtaAll"]->Fill( jet_it->eta(), weight );

    if (njets == 1) { hists_["jetPt1" ]->Fill( jet_it->pt() , weight );
                      hists_["jetEta1"]->Fill( jet_it->eta(), weight );
                    }
    if (njets == 2) { hists_["jetPt2" ]->Fill( jet_it->pt() , weight );
                      hists_["jetEta2"]->Fill( jet_it->eta(), weight );
                    }
    if (njets == 3) { hists_["jetPt3" ]->Fill( jet_it->pt() , weight );
                      hists_["jetEta3"]->Fill( jet_it->eta(), weight );
                    }
    if (njets == 4) { hists_["jetPt4" ]->Fill( jet_it->pt() , weight );
                      hists_["jetEta4"]->Fill( jet_it->eta(), weight );
                    }
  }

}


// ------------ method called once each job just before starting event loop  ------------
void 
TTbar_GenJetAnalyzer::beginJob()
{
  if(!dbe) return;
  dbe->setCurrentFolder("Generator/TTbar");
  hists_["jetPtAll" ] = dbe->book1D("TTbar_jetPtAll" , "pt" , 1000,  0., 1000.); 
  hists_["jetPt1"   ] = dbe->book1D("TTbar_jetPt1"   , "pt" , 1000,  0., 1000.); 
  hists_["jetPt2"   ] = dbe->book1D("TTbar_jetPt2"   , "pt" , 1000,  0., 1000.); 
  hists_["jetPt3"   ] = dbe->book1D("TTbar_jetPt3"   , "pt" , 1000,  0., 1000.); 
  hists_["jetPt4"   ] = dbe->book1D("TTbar_jetPt4"   , "pt" , 1000,  0., 1000.); 
                                                                                    
  hists_["jetEtaAll"] = dbe->book1D("TTbar_jetEtaAll", "eta",  100, -5.,    5.); 
  hists_["jetEta1"  ] = dbe->book1D("TTbar_jetEta1"  , "eta",  100, -5.,    5.); 
  hists_["jetEta2"  ] = dbe->book1D("TTbar_jetEta2"  , "eta",  100, -5.,    5.); 
  hists_["jetEta3"  ] = dbe->book1D("TTbar_jetEta3"  , "eta",  100, -5.,    5.); 
  hists_["jetEta4"  ] = dbe->book1D("TTbar_jetEta4"  , "eta",  100, -5.,    5.); 
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TTbar_GenJetAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
TTbar_GenJetAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
TTbar_GenJetAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
TTbar_GenJetAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
TTbar_GenJetAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TTbar_GenJetAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

