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

  int lineWidth = 75;
  if( jets->begin()->isCaloJet() )
    lineWidth = 100;
  else if( jets->begin()->isPFJet() )
    lineWidth = 120;

  std::cout << std::setfill('=') << std::setw(lineWidth) << "\n" << std::setfill(' ');
  std::cout << std::setw( 5) << "jet :"
            << std::setw(11) << "pt :"
            << std::setw( 9) << "eta :"
            << std::setw( 9) << "phi :"
	    << std::setw(11) << "TCHE :"
            << std::setw(11) << "TCHP :"
            << std::setw( 9) << "SSVHE :"
            << std::setw( 9) << "SSVHP :";
  if( jets->begin()->isCaloJet() ) {
    std::cout << std::setw( 8) << "emf :"
	      << std::setw(10) << "n90Hits :"
	      << std::setw( 7) << "fHPD";
  }
  if( jets->begin()->isPFJet() ) {
    std::cout << std::setw(9) << "chf : "
	      << std::setw(8) << "nhf : "
	      << std::setw(8) << "cef : "
	      << std::setw(8) << "nef : "
	      << std::setw(6) << "nCh : "
	      << std::setw(6) << "nConst";
  }
  std::cout << std::endl
	    << std::setfill('-') << std::setw(lineWidth) << "\n" << std::setfill(' ');
  unsigned i=0;
  for(std::vector<pat::Jet>::const_iterator jet=jets->begin(); jet!=jets->end(); ++jet){
    std::cout << std::setw(3) << i << " : " << std::setprecision(3) << std::fixed
	      << std::setw(8) << jet->pt() << " : "
	      << std::setw(6) << jet->eta() << " : "
	      << std::setw(6) << jet->phi() << " : "
	      << std::setw(8) << jet->bDiscriminator("trackCountingHighEffBJetTags") << " : "
	      << std::setw(8) << jet->bDiscriminator("trackCountingHighPurBJetTags") << " : "
	      << std::setw(6) << jet->bDiscriminator("simpleSecondaryVertexHighEffBJetTags") << " : "
	      << std::setw(6) << jet->bDiscriminator("simpleSecondaryVertexHighPurBJetTags") << " : ";
    if( jet->isCaloJet() ) {
      std::cout << std::setw(5) << jet->emEnergyFraction() << " : "
		<< std::setw(7) << jet->jetID().n90Hits << " : "
		<< std::setw(6) << jet->jetID().fHPD;
    }
    if( jet->isPFJet() ) {
      std::cout << std::setw(5) << jet->chargedHadronEnergyFraction() << " : "
		<< std::setw(5) << jet->neutralHadronEnergyFraction() << " : "
		<< std::setw(5) << jet->chargedEmEnergyFraction() << " : "
		<< std::setw(5) << jet->neutralEmEnergyFraction() << " : "
		<< std::setw(3) << jet->chargedMultiplicity() << " : "
		<< std::setw(6) << jet->nConstituents();
    }
    std::cout << std::endl;
    i++;
  }
  std::cout << std::setfill('=') << std::setw(lineWidth) << "\n" << std::setfill(' ');
}

void TopJetAnalyzer::beginJob()
{
}

void TopJetAnalyzer::endJob()
{
}

  
