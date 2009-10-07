#include "DataFormats/PatCandidates/interface/Jet.h"
#include "TopQuarkAnalysis/Examples/plugins/TopJetAnalyzer.h"


TopJetAnalyzer::TopJetAnalyzer(const edm::ParameterSet& cfg):
  input_(cfg.getParameter<edm::InputTag>("input"))
{
  edm::Service<TFileService> fs;
  
  Num_Jets   = fs->make<TH1I>("Number_of_Jets","Num_{Jets}",    10,  0 ,  10 );
  pt_Jets    = fs->make<TH1F>("pt_of_Jets",    "pt_{Jets}",    100,  0., 300.);
  energy_Jets=fs->make<TH1F> ("energy_of_Jets","energy_{Jets}",100,  0., 300.);
  eta_Jets   =fs->make<TH1F> ("eta_of_Jets",   "eta_{Jets}",   100, -3.,   3.);
  phi_Jets   =fs->make<TH1F> ("phi_of_Jets",   "phi_{Jets}",   100, -4.,   4.);

  btag_Jets  =fs->make<TH1F> ("btag_of_Jets",  "btag_{Jet}",   400,-20.,  20.);
}

TopJetAnalyzer::~TopJetAnalyzer()
{
}

void
TopJetAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<std::vector<pat::Jet> > jets;
  evt.getByLabel(input_, jets); 
  
  Num_Jets->Fill( jets->size());
  for( std::vector<pat::Jet>::const_iterator jet=jets->begin(); 
       jet!=jets->end(); ++jet){
    pt_Jets    ->Fill( jet->pt()    );
    energy_Jets->Fill( jet->energy());
    eta_Jets   ->Fill( jet->eta()   );
    phi_Jets   ->Fill( jet->phi()   );

    btag_Jets  ->Fill( jet->bDiscriminator("combinedSecondaryVertexBJetTags") );
  }    
}

void TopJetAnalyzer::beginJob(const edm::EventSetup&)
{
}

void TopJetAnalyzer::endJob()
{
}

  
