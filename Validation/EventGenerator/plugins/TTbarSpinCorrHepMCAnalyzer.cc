#include "Validation/EventGenerator/interface/TTbarSpinCorrHepMCAnalyzer.h"

//
// constructors and destructor
//
TTbarSpinCorrHepMCAnalyzer::TTbarSpinCorrHepMCAnalyzer(const edm::ParameterSet& iConfig) :
  genEventInfoProductTag_(iConfig.getParameter<edm::InputTag>("genEventInfoProductTag")),
  genParticlesTag_(iConfig.getParameter<edm::InputTag>("genParticlesTag"))
{
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
}


TTbarSpinCorrHepMCAnalyzer::~TTbarSpinCorrHepMCAnalyzer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
TTbarSpinCorrHepMCAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // --- the MC weights ---
  Handle<GenEventInfoProduct> evt_info;
  iEvent.getByLabel(genEventInfoProductTag_, evt_info);
  if (evt_info.failedToGet())
    return;

  weight = evt_info->weight() ;
  
  // --- get genParticles ---
  Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByLabel(genParticlesTag_, genParticles);   

  const reco::GenParticle * _lepton   (0) ;
  const reco::GenParticle * _leptonBar(0) ;

  bool hasTop(false), hasTopbar(false);
  for(size_t i = 0; i < genParticles->size(); ++ i) {
    const reco::GenParticle & p = (*genParticles)[i];
    if(p.pdgId() == 6) hasTop=true;
    if(p.pdgId() == -6) hasTopbar=true;
  }

  if(hasTop && hasTopbar){
    // --- get status 3 leptons
    for(size_t i = 0; i < genParticles->size(); ++ i) {
      const reco::GenParticle & p = (*genParticles)[i];
      if ( (p.pdgId() ==  11 ||
	    p.pdgId() ==  13 ||
	    p.pdgId() ==  15) && p.status() == 3) { _lepton    = &p ; }
      if ( (p.pdgId() == -11 ||
	    p.pdgId() == -13 ||
	    p.pdgId() == -15) && p.status() == 3) { _leptonBar = &p ; }
      
      if (_lepton && _leptonBar) break;
    }
    
    if (_lepton && _leptonBar) {
      
      math::XYZTLorentzVector lepton    = _lepton   ->p4() ;
      math::XYZTLorentzVector leptonBar = _leptonBar->p4() ;
      
      double deltaPhi = fabs(TVector2::Phi_mpi_pi(lepton.phi() - leptonBar.phi())) ;
      _h_deltaPhi->Fill(deltaPhi, weight) ;
      
      double asym = ( deltaPhi > CLHEP::halfpi ) ? 0.5 : -0.5 ;
      _h_asym->Fill(asym, weight) ;
      
      math::XYZTLorentzVector llpair    = lepton + leptonBar ;
      
      double llpairPt = llpair.pt() ;
      _h_llpairPt->Fill(llpairPt, weight) ;
      
      double llpairM  = llpair.M() ;
    _h_llpairM ->Fill(llpairM , weight) ;
    
    }
    nEvt->Fill(0.5 , weight) ;
  }
}


// ------------ method called once each job just before starting event loop  ------------
void 
TTbarSpinCorrHepMCAnalyzer::beginJob()
{
  if(dbe){
    ///Setting the DQM top directories
    TString dir="Generator/";
    dir+="TTbarSpinCorr";
    dbe->setCurrentFolder(dir.Data());

    // Number of analyzed events
    nEvt = dbe->book1D("nEvt", "n analyzed Events", 1, 0., 1.);
    
    _h_asym = dbe->book1D("TTbar_asym","Asymmetr", 2, -1., 1.);
    _h_asym->setAxisTitle("Asymmetry");

    _h_deltaPhi = dbe->book1D("TTbar_deltaPhi","#Delta#phi(ll)", 320, 0, 3.2);
    _h_deltaPhi->setAxisTitle("#Delta#phi(ll)");
    
    _h_llpairPt = dbe->book1D("TTbar_llpairPt","Lepton pair transverse momentum", 1000, 0, 1000);
    _h_llpairPt->setAxisTitle("p_{T}(ll)");
    
    _h_llpairM  = dbe->book1D("TTbar_llpairM","Lepton pair invariant mass", 1000, 0, 1000);
    _h_llpairM->setAxisTitle("M(ll)");

  }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TTbarSpinCorrHepMCAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
TTbarSpinCorrHepMCAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
TTbarSpinCorrHepMCAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
TTbarSpinCorrHepMCAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
TTbarSpinCorrHepMCAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(TTbarSpinCorrHepMCAnalyzer);
