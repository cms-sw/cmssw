#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TopQuarkAnalysis/Examples/plugins/TopJetAnalyzer.h"


TopJetAnalyzer::TopJetAnalyzer(const edm::ParameterSet& cfg):
  input_(cfg.getParameter<edm::InputTag>("input"))
{
  edm::Service<TFileService> fs;
  
  Num_Jets   = fs->make<TH1I>("Number_of_Jets","Num_{Jets}",    10,  0 ,  10 );
  pt_Jets    = fs->make<TH1F>("pt_of_Jets",    "pt_{Jets}",    100,  0., 300.);
  energy_Jets=fs->make<TH1F> ("energy_of_Jets","energy_{Jets}",100,  0., 300.);
  eta_Jets   =fs->make<TH1F> ("eta_of_Jets",   "eta_{Jets}",   100, -3.,   3.);
  phi_Jets   =fs->make<TH1F> ("phi_of_Jets",   "phi_{Jets}",   100, -5.,   5.);
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
  for( std::vector<pat::Jet>::const_iterator jet=jets->begin(); jet!=jets->end(); ++jet){
    pt_Jets    ->Fill( jet->pt()    );
    energy_Jets->Fill( jet->energy());
    eta_Jets   ->Fill( jet->eta()   );
    phi_Jets   ->Fill( jet->phi()   );

//     // test JEC from PAT
//     if(jet == jets->begin()){
//     edm::LogVerbatim log("TopJetAnalyzer_jec");
//     //jet->jetCorrFactors().print();
//     log << "--------------------------------\n";
//     log << " Jet Energy Correction Factors: \n";
//     log << "--------------------------------\n";
//     // uncomment for use with PATv1
//     // log << "  " << jet->jetCorrName() << ": " << jet->pt() << " (default) \n";
//     // uncomment for use with PATv2
//     log << "  " << jet->corrStep() << ": " << jet->pt() << " (default) \n";
//     log << "--------------------------------\n";
//     log << "  " << jet->correctedJet("raw")        .jetCorrName() << ": " << jet->correctedJet("raw")        .pt() << "\n";
//     log << "  " << jet->correctedJet("off")        .jetCorrName() << ": " << jet->correctedJet("off")        .pt() << "\n";
//     log << "  " << jet->correctedJet("rel")        .jetCorrName() << ": " << jet->correctedJet("rel")        .pt() << "\n";
//     log << "  " << jet->correctedJet("abs")        .jetCorrName() << ": " << jet->correctedJet("abs")        .pt() << "\n";
//     log << "  " << jet->correctedJet("emf")        .jetCorrName() << ": " << jet->correctedJet("emf")        .pt() << "\n";
//     log << "  " << jet->correctedJet("had",  "glu").jetCorrName() << ": " << jet->correctedJet("had",  "glu").pt() << " (gluon )\n";
//     log << "  " << jet->correctedJet("had",  "uds").jetCorrName() << ": " << jet->correctedJet("had",  "uds").pt() << " (uds   )\n";
//     log << "  " << jet->correctedJet("had",  "c"  ).jetCorrName() << ": " << jet->correctedJet("had",  "c"  ).pt() << " (charm )\n";
//     log << "  " << jet->correctedJet("had",  "b"  ).jetCorrName() << ": " << jet->correctedJet("had",  "b"  ).pt() << " (beauty)\n";
//     log << "  " << jet->correctedJet("ue",   "glu").jetCorrName() << ": " << jet->correctedJet("ue",   "glu").pt() << " (gluon )\n";
//     log << "  " << jet->correctedJet("ue",   "uds").jetCorrName() << ": " << jet->correctedJet("ue",   "uds").pt() << " (uds   )\n";
//     log << "  " << jet->correctedJet("ue",   "c"  ).jetCorrName() << ": " << jet->correctedJet("ue",   "c"  ).pt() << " (charm )\n";
//     log << "  " << jet->correctedJet("ue",   "b"  ).jetCorrName() << ": " << jet->correctedJet("ue",   "b"  ).pt() << " (beauty)\n";
//     log << "  " << jet->correctedJet("part", "glu").jetCorrName() << ": " << jet->correctedJet("part", "glu").pt() << " (gluon )\n";
//     log << "  " << jet->correctedJet("part", "uds").jetCorrName() << ": " << jet->correctedJet("part", "uds").pt() << " (uds   )\n";
//     log << "  " << jet->correctedJet("part", "c"  ).jetCorrName() << ": " << jet->correctedJet("part", "c"  ).pt() << " (charm )\n";
//     log << "  " << jet->correctedJet("part", "b"  ).jetCorrName() << ": " << jet->correctedJet("part", "b"  ).pt() << " (beauty)\n";    
//    }
    btag_Jets  ->Fill( jet->bDiscriminator("combinedSecondaryVertexBJetTags") );
  }    
}

void TopJetAnalyzer::beginJob(const edm::EventSetup&)
{
}

void TopJetAnalyzer::endJob()
{
}

  
