#include "DataFormats/PatCandidates/interface/Muon.h"
#include "TopQuarkAnalysis/Examples/plugins/TopMuonAnalyzer.h"


TopMuonAnalyzer::TopMuonAnalyzer(const edm::ParameterSet& cfg):
  inputToken_  (consumes<std::vector<pat::Muon> >(cfg.getParameter<edm::InputTag>("input"  ))),
  verbose_(cfg.getParameter<bool>         ("verbose"))
{
  edm::Service<TFileService> fs;

  mult_ = fs->make<TH1F>("mult", "multiplicity (muons)", 10,  0 ,   10);
  en_   = fs->make<TH1F>("en"  , "energy (muons)",       60,  0., 300.);
  pt_   = fs->make<TH1F>("pt"  , "pt (muons)",           60,  0., 300.);
  eta_  = fs->make<TH1F>("eta" , "eta (muons)",          30, -3.,   3.);
  phi_  = fs->make<TH1F>("phi" , "phi (muons)",          40, -4.,   4.);

}

TopMuonAnalyzer::~TopMuonAnalyzer()
{
}

void
TopMuonAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<std::vector<pat::Muon> > muons;
  evt.getByToken(inputToken_, muons);

  // fill histograms

  mult_->Fill( muons->size() );
  for(std::vector<pat::Muon>::const_iterator muon=muons->begin(); muon!=muons->end(); ++muon){
    pt_ ->Fill( muon->pt()     );
    en_ ->Fill( muon->energy() );
    eta_->Fill( muon->eta()    );
    phi_->Fill( muon->phi()    );
  }

  // produce printout if desired

  if( muons->empty() || !verbose_ )
    return;

  unsigned i=0;

  std::cout << "==================================================================="
            << std::endl;
  std::cout << std::setw(5 ) << "mu  :"
            << std::setw(13) << "pt :"
            << std::setw(13) << "eta :"
            << std::setw(13) << "phi :"
	    << std::setw(13) << "relIso :"
	    << std::setw(6 ) << "GLB :"
	    << std::setw(4 ) << "TRK" << std::endl;
  std::cout << "-------------------------------------------------------------------"
            << std::endl;
  for(std::vector<pat::Muon>::const_iterator muon=muons->begin(); muon!=muons->end(); ++muon){
    std::cout << std::setw(3 ) << i << " : "
	      << std::setw(10) << muon->pt() << " : "
	      << std::setw(10) << muon->eta() << " : "
	      << std::setw(10) << muon->phi() << " : "
	      << std::setw(10) << (muon->trackIso()+muon->caloIso())/muon->pt() << " : "
	      << std::setw( 3) << muon->isGlobalMuon() << " : "
      	      << std::setw( 3) << muon->isTrackerMuon() << std::endl;
    i++;
  }
  std::cout << "==================================================================="
            << std::endl;
}

void TopMuonAnalyzer::beginJob()
{
}

void TopMuonAnalyzer::endJob()
{
}
