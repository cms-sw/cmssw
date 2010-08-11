#include "DataFormats/PatCandidates/interface/Jet.h"
#include "TopQuarkAnalysis/Examples/plugins/TopJetAnalyzer.h"


TopJetAnalyzer::TopJetAnalyzer(const edm::ParameterSet& cfg):
  input_  (cfg.getParameter<edm::InputTag>("input"  )),
  verbose_(cfg.getParameter<bool>         ("verbose"))
{
  edm::Service<TFileService> fs;
  
  mult_ = fs->make<TH1F>("mult", "multiplicity (jets)", 30,  0 ,   30);
  en_   = fs->make<TH1F>("en"  , "energy (jets)",       60,  0., 300.);
  pt_   = fs->make<TH1F>("pt"  , "pt (jets)",           60,  0., 300.);
  eta_  = fs->make<TH1F>("eta" , "eta (jets)",          30, -3.,   3.);
  phi_  = fs->make<TH1F>("phi" , "phi (jets)",          40, -4.,   4.);
}

TopJetAnalyzer::~TopJetAnalyzer()
{
}

void
TopJetAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<std::vector<pat::Jet> > jets;
  evt.getByLabel(input_, jets); 

  // fill histograms
  
  mult_->Fill( jets->size() );
  for(std::vector<pat::Jet>::const_iterator jet=jets->begin(); jet!=jets->end(); ++jet){
    pt_ ->Fill( jet->pt()     );
    en_ ->Fill( jet->energy() );
    eta_->Fill( jet->eta()    );
    phi_->Fill( jet->phi()    );
  }

  // produce printout if desired

  if( jets->size()<1 || !verbose_ )
    return;

  std::cout << "================================================================="
            << "================================================================="
            << std::endl;
  std::cout << std::setw(5 ) << "jet :"
            << std::setw(13) << "pt :"
            << std::setw(13) << "eta :"
            << std::setw(13) << "phi :"
	    << std::setw(13) << "emf :"
	    << std::setw(10) << "n90Hits :"
	    << std::setw(13) << "fHPD :"
            << std::setw(13) << "TCHE :"
            << std::setw(13) << "TCHP :"
            << std::setw(13) << "SSVHE :"
            << std::setw(11) << "SSVHP" << std::endl;
  std::cout << "-----------------------------------------------------------------"
            << "-----------------------------------------------------------------"
            << std::endl;
  unsigned i=0;
  for(std::vector<pat::Jet>::const_iterator jet=jets->begin(); jet!=jets->end(); ++jet){
    std::cout << std::setw(3 ) << i << " : "
	      << std::setw(10) << jet->pt() << " : "
	      << std::setw(10) << jet->eta() << " : "
	      << std::setw(10) << jet->phi() << " : "
              << std::setw(10) << jet->emEnergyFraction() << " : "
              << std::setw(7 ) << jet->jetID().n90Hits << " : "
              << std::setw(10) << jet->jetID().fHPD << " : "
	      << std::setw(10) << jet->bDiscriminator("trackCountingHighEffBJetTags") << " : "
	      << std::setw(10) << jet->bDiscriminator("trackCountingHighPurBJetTags") << " : "
	      << std::setw(10) << jet->bDiscriminator("simpleSecondaryVertexHighEffBJetTags") << " : "
	      << std::setw(10) << jet->bDiscriminator("simpleSecondaryVertexHighPurBJetTags") << std::endl;
    i++;
  }
  std::cout << "================================================================="
            << "================================================================="
            << std::endl;
}

void TopJetAnalyzer::beginJob()
{
}

void TopJetAnalyzer::endJob()
{
}

  
