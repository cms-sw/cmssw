#include "Validation/EventGenerator/interface/TTbar_GenJetAnalyzer.h"


TTbar_GenJetAnalyzer::TTbar_GenJetAnalyzer(const edm::ParameterSet& iConfig):
  jets_(iConfig.getParameter<edm::InputTag>("jets")),
  genEventInfoProductTag_(iConfig.getParameter<edm::InputTag>("genEventInfoProductTag"))
{

  genEventInfoProductTagToken_=consumes<GenEventInfoProduct>(genEventInfoProductTag_);
  jetsToken_=consumes<std::vector<reco::GenJet> >(jets_);

}


TTbar_GenJetAnalyzer::~TTbar_GenJetAnalyzer(){}


void TTbar_GenJetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
 
  // --- the MC weights ---
  Handle<GenEventInfoProduct> evt_info;
  iEvent.getByToken(genEventInfoProductTagToken_, evt_info);
  if(!evt_info.isValid()) return;
  weight = evt_info->weight() ;

  // Gather information in the GenJet collection
  edm::Handle<std::vector<reco::GenJet> > jets;
  iEvent.getByToken(jetsToken_, jets );

  if(!jets.isValid()) return;
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


void TTbar_GenJetAnalyzer::bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &){
  i.setCurrentFolder("Generator/TTbar");
  hists_["jetPtAll" ] = i.book1D("TTbar_jetPtAll" , "pt" , 1000,  0., 1000.); 
  hists_["jetPt1"   ] = i.book1D("TTbar_jetPt1"   , "pt" , 1000,  0., 1000.); 
  hists_["jetPt2"   ] = i.book1D("TTbar_jetPt2"   , "pt" , 1000,  0., 1000.); 
  hists_["jetPt3"   ] = i.book1D("TTbar_jetPt3"   , "pt" , 1000,  0., 1000.); 
  hists_["jetPt4"   ] = i.book1D("TTbar_jetPt4"   , "pt" , 1000,  0., 1000.); 
                                                                                    
  hists_["jetEtaAll"] = i.book1D("TTbar_jetEtaAll", "eta",  100, -5.,    5.); 
  hists_["jetEta1"  ] = i.book1D("TTbar_jetEta1"  , "eta",  100, -5.,    5.); 
  hists_["jetEta2"  ] = i.book1D("TTbar_jetEta2"  , "eta",  100, -5.,    5.); 
  hists_["jetEta3"  ] = i.book1D("TTbar_jetEta3"  , "eta",  100, -5.,    5.); 
  hists_["jetEta4"  ] = i.book1D("TTbar_jetEta4"  , "eta",  100, -5.,    5.); 
}

