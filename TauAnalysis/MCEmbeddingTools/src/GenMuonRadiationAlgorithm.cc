#include "TauAnalysis/MCEmbeddingTools/interface/GenMuonRadiationAlgorithm.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "GeneratorInterface/ExternalDecays/interface/DecayRandomEngine.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "HepMC/IO_HEPEVT.h"

bool GenMuonRadiationAlgorithm::photos_isInitialized_ = false;

GenMuonRadiationAlgorithm::GenMuonRadiationAlgorithm(const edm::ParameterSet& cfg)
  : beamEnergy_(cfg.getParameter<double>("beamEnergy")),
    photos_(cfg.getParameter<edm::ParameterSet>("PhotosOptions"))
{
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( !rng.isAvailable() ) 
    throw cms::Exception("Configuration")
      << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
      << "which appears to be absent. Please add that service to your configuration\n"
      << "or remove the modules that require it.\n";
  
  // this is a global variable defined in GeneratorInterface/ExternalDecays/src/ExternalDecayDriver.cc
  decayRandomEngine = &rng->getEngine();

  verbosity_ = ( cfg.exists("verbosity") ) ? 
    cfg.getParameter<int>("verbosity") : 0;
}

namespace
{
  double square(double x)
  {
    return x*x;
  }
}

namespace
{
  reco::Candidate::LorentzVector getP4_limited(const reco::Candidate::LorentzVector& p4, double mass)
  {
    // CV: restrict reconstructed momenta to 1 TeV maximum
    //    (higher values are almost for sure due to reconstruction errors)
    reco::Candidate::LorentzVector p4_limited = p4;
    if ( p4.energy() > 1.e+3 ) {
      double scaleFactor = 1.e+3/p4.energy();
      double px_limited = scaleFactor*p4.px();
      double py_limited = scaleFactor*p4.py();
      double pz_limited = scaleFactor*p4.pz();
      double en_limited = sqrt(square(px_limited) + square(py_limited) + square(pz_limited) + square(mass));
      p4_limited.SetPxPyPzE(
        px_limited, 
	py_limited, 
	pz_limited, 
	en_limited);
    }
    return p4_limited;
  }
}

reco::Candidate::LorentzVector GenMuonRadiationAlgorithm::compFSR(const reco::Candidate::LorentzVector& muonP4, int muonCharge,
								  const reco::Candidate::LorentzVector& otherP4)
{
  if ( verbosity_ ) {
    std::cout << "<GenMuonRadiationAlgorithm::compMuonFSR>:" << std::endl;
    std::cout << " muon: En = " << muonP4.E() << ", Pt = " << muonP4.pt() << ", eta = " << muonP4.eta() << ", phi = " << muonP4.phi() << ", charge = " << muonCharge << std::endl;
    std::cout << " other: En = " << otherP4.E() << ", Pt = " << otherP4.pt() << ", eta = " << otherP4.eta() << ", phi = " << otherP4.phi() << std::endl;
  }

  if ( !photos_isInitialized_ ) {
    photos_.init();
    photos_isInitialized_ = true;
  }

  HepMC::GenEvent* genEvt_beforeFSR = new HepMC::GenEvent();

  HepMC::GenVertex* genVtx_out = new HepMC::GenVertex(HepMC::FourVector(0.,0.,0.,0.));

  reco::Candidate::LorentzVector muonP4_limited = getP4_limited(muonP4, muonP4.mass());
  
  reco::Candidate::LorentzVector otherP4_limited = getP4_limited(otherP4, 0.);
  reco::Candidate::LorentzVector sumP4 = muonP4_limited + otherP4_limited;
  if ( verbosity_ ) {
    std::cout << "muon(limited): En = " << muonP4_limited.E() << ", Pt = " << muonP4_limited.pt() << ", eta = " << muonP4_limited.eta() << ", phi = " << muonP4_limited.phi() << std::endl;
    std::cout << "other: En = " << otherP4_limited.E() << ", Pt = " << otherP4_limited.pt() << ", eta = " << otherP4_limited.eta() << ", phi = " << otherP4_limited.phi() << std::endl;
    std::cout << "W': En = " << sumP4.E() << ", Pt = " << sumP4.pt() << ", eta = " << sumP4.eta() << ", phi = " << sumP4.phi() << std::endl;
  }
  // CV: use pdgId code for W',
  //     so that muon + neutrino system may have arbitrary mass
  //int wPdgId = +34*muonCharge;
  //int wPdgId = 32;
  int wPdgId = 23;
  HepMC::GenParticle* genW = new HepMC::GenParticle((HepMC::FourVector)sumP4, wPdgId, 2, HepMC::Flow(), HepMC::Polarization(0,0));
  genVtx_out->add_particle_in(genW);

  int muonPdgId = -13*muonCharge;
  HepMC::GenParticle* genMuon = new HepMC::GenParticle((HepMC::FourVector)muonP4_limited, muonPdgId, 1, HepMC::Flow(), HepMC::Polarization(0,0));
  genVtx_out->add_particle_out(genMuon);

  //int neutrinoPdgId = +14*muonCharge;
  int neutrinoPdgId = +13*muonCharge;
  HepMC::GenParticle* genNeutrino = new HepMC::GenParticle((HepMC::FourVector)otherP4_limited, neutrinoPdgId, 1, HepMC::Flow(), HepMC::Polarization(0,0));
  genVtx_out->add_particle_out(genNeutrino);

  genEvt_beforeFSR->add_vertex(genVtx_out);

  if ( verbosity_ ) {
    std::cout << "genEvt (beforeFSR):" << std::endl;
    genEvt_beforeFSR->print(std::cout);
  }

  HepMC::IO_HEPEVT conv;
  conv.write_event(genEvt_beforeFSR); 
  HepMC::GenEvent* genEvt_afterFSR = photos_.apply(genEvt_beforeFSR);
  if ( verbosity_ ) {
    std::cout << "genEvt (afterFSR):" << std::endl;
    genEvt_afterFSR->print(std::cout);
  }
  
  // find lowest energy muon
  // (= muon after FSR) 
  reco::Candidate::LorentzVector muonP4afterFSR = muonP4;
  for ( HepMC::GenEvent::particle_iterator genParticle = genEvt_afterFSR->particles_begin();
	genParticle != genEvt_afterFSR->particles_end(); ++genParticle ) {
    if ( (*genParticle)->pdg_id() == muonPdgId && (*genParticle)->momentum().e() < muonP4afterFSR.E() ) {
      muonP4afterFSR.SetPxPyPzE(
	(*genParticle)->momentum().px(), 
	(*genParticle)->momentum().py(), 
	(*genParticle)->momentum().pz(), 
	(*genParticle)->momentum().e());
    }	 
  }
  
  reco::Candidate::LorentzVector sumPhotonP4 = muonP4 - muonP4afterFSR;
  if ( verbosity_ ) {
    std::cout << "sumPhotons: En = " << sumPhotonP4.E() << ", Pt = " << sumPhotonP4.pt() << ", eta = " << sumPhotonP4.eta() << ", phi = " << sumPhotonP4.phi() << std::endl;
    if ( sumPhotonP4.E() > 80. ) std::cout << "--> CHECK !!" << std::endl;
  }

  delete genEvt_beforeFSR;
  if ( genEvt_afterFSR != genEvt_beforeFSR ) delete genEvt_afterFSR;

  return sumPhotonP4;
}
