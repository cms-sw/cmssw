#include "DataFormats/PatCandidates/interface/Electron.h"
#include "TopQuarkAnalysis/Examples/plugins/TopElecAnalyzer.h"

TopElecAnalyzer::TopElecAnalyzer(const edm::ParameterSet& cfg):
  input_  (cfg.getParameter<edm::InputTag>("input"  )),
  verbose_(cfg.getParameter<bool>         ("verbose"))
{
  edm::Service<TFileService> fs;
  
  mult_ = fs->make<TH1F>("mult", "multiplicity (electrons)", 10,  0 ,   10);
  en_   = fs->make<TH1F>("en"  , "energy (electrons)"      , 60,  0., 300.);
  pt_   = fs->make<TH1F>("pt"  , "pt (electrons)"          , 60,  0., 300.);
  eta_  = fs->make<TH1F>("eta" , "eta (electrons)"         , 30, -3.,   3.);
  phi_  = fs->make<TH1F>("phi" , "phi (electrons)"         , 40, -4.,   4.);
}

TopElecAnalyzer::~TopElecAnalyzer()
{
}

void
TopElecAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{       
  edm::Handle<std::vector<pat::Electron> > elecs;
  evt.getByLabel(input_, elecs); 

  // fill histograms

  mult_->Fill( elecs->size() );
  for(std::vector<pat::Electron>::const_iterator elec=elecs->begin(); elec!=elecs->end(); ++elec){
    en_ ->Fill( elec->energy() );
    pt_ ->Fill( elec->pt()     );
    eta_->Fill( elec->eta()    );
    phi_->Fill( elec->phi()    );
  }

  // produce printout if desired

  if( elecs->size()<1 || !verbose_ )
    return;

  unsigned i=0;

  std::cout << "======================================================="
            << std::endl;
  std::cout << std::setw(5 ) << "ele :"
            << std::setw(13) << "et :"
            << std::setw(13) << "eta :"
            << std::setw(13) << "phi :"
	    << std::setw(11) << "relIso" << std::endl;
  std::cout << "-------------------------------------------------------"
            << std::endl;
  for(std::vector<pat::Electron>::const_iterator elec=elecs->begin(); elec!=elecs->end(); ++elec){
    std::cout << std::setw(3 ) << i << " : "
	      << std::setw(10) << elec->pt() << " : "
	      << std::setw(10) << elec->eta() << " : "
	      << std::setw(10) << elec->phi() << " : "
	      << std::setw(10) << (elec->dr03TkSumPt()+elec->dr03EcalRecHitSumEt()+elec->dr03HcalTowerSumEt())/elec->et() << std::endl;
    i++;
  }
  std::cout << "======================================================="
            << std::endl;
}

void TopElecAnalyzer::beginJob()
{
}

void TopElecAnalyzer::endJob()
{
}
  
