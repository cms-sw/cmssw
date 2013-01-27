#include "TauAnalysis/MCEmbeddingTools/interface/GenMuonRadiationAlgorithm.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "GeneratorInterface/ExternalDecays/interface/DecayRandomEngine.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Declarations.h" // needed for call_pyhepc
#include "TauAnalysis/MCEmbeddingTools/interface/extraPythia.h"                // needed for call_pyexec

#include "DataFormats/Candidate/interface/Candidate.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include "HepMC/IO_HEPEVT.h"
#include "HepPID/ParticleIDTranslations.hh"

#include <Math/VectorUtil.h>

bool GenMuonRadiationAlgorithm::photos_isInitialized_ = false;
bool GenMuonRadiationAlgorithm::pythia_isInitialized_ = false;

GenMuonRadiationAlgorithm::GenMuonRadiationAlgorithm(const edm::ParameterSet& cfg)
  : beamEnergy_(cfg.getParameter<double>("beamEnergy")),
    photos_(0),
    pythia_(0)
{
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( !rng.isAvailable() ) 
    throw cms::Exception("Configuration")
      << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
      << "which appears to be absent. Please add that service to your configuration\n"
      << "or remove the modules that require it.\n";
  
  // this is a global variable defined in GeneratorInterface/ExternalDecays/src/ExternalDecayDriver.cc
  decayRandomEngine = &rng->getEngine();

  std::string mode_string = cfg.getParameter<std::string>("mode");
  if      ( mode_string == "pythia" ) mode_ = kPYTHIA;
  else if ( mode_string == "photos" ) mode_ = kPHOTOS;
  else throw cms::Exception("Configuration")
    << " Invalid Configuration Parameter 'mode' = " << mode_string << " !!\n";

  if ( mode_ == kPYTHIA ) pythia_ = new gen::Pythia6Service(cfg);
  if ( mode_ == kPHOTOS ) photos_ = new gen::PhotosInterface(cfg.getParameter<edm::ParameterSet>("PhotosOptions"));

  verbosity_ = ( cfg.exists("verbosity") ) ? 
    cfg.getParameter<int>("verbosity") : 0;
}

GenMuonRadiationAlgorithm::~GenMuonRadiationAlgorithm()
{
  delete photos_;
  call_pystat(1);
  delete pythia_;
}

namespace
{
  double square(double x)
  {
    return x*x;
  }

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

  void printPYJETS()
  {
    int numParticles = pyjets.n;
    for ( int iParticle = 0; iParticle < numParticles; ++iParticle ) {
      reco::Candidate::LorentzVector particleP4(
	pyjets.p[0][iParticle],
	pyjets.p[1][iParticle],
	pyjets.p[2][iParticle],
	pyjets.p[3][iParticle]);
      std::cout << " particle #" << iParticle << " (pythiaId = " << pyjets.k[1][iParticle] << ", status = " << pyjets.k[0][iParticle] << "):"
		<< " Pt = " << particleP4.pt() << ", eta = " << particleP4.eta() << ", phi = " << particleP4.phi() << ", mass = " << particleP4.mass() << std::endl;
    }
  }
}

reco::Candidate::LorentzVector GenMuonRadiationAlgorithm::compFSR(const reco::Candidate::LorentzVector& muonP4, int muonCharge,
								  const reco::Candidate::LorentzVector& otherP4, int& errorFlag)
{
  if ( verbosity_ ) {
    std::cout << "<GenMuonRadiationAlgorithm::compMuonFSR>:" << std::endl;
    std::cout << " muon: En = " << muonP4.E() << ", Pt = " << muonP4.pt() << ", eta = " << muonP4.eta() << ", phi = " << muonP4.phi() << ", charge = " << muonCharge << std::endl;
    std::cout << " other: En = " << otherP4.E() << ", Pt = " << otherP4.pt() << ", eta = " << otherP4.eta() << ", phi = " << otherP4.phi() << std::endl;
  }

  reco::Candidate::LorentzVector muonP4_limited = getP4_limited(muonP4, muonP4.mass());  
  int muonPdgId = -13*muonCharge;
  reco::Candidate::LorentzVector otherP4_limited = getP4_limited(otherP4, otherP4.mass());
  int otherPdgId = +13*muonCharge;
  reco::Candidate::LorentzVector zP4 = muonP4_limited + otherP4_limited;
  int zPdgId = 23;
  if ( verbosity_ ) {
    std::cout << "muon(limited): En = " << muonP4_limited.E() << ", Pt = " << muonP4_limited.pt() << ", eta = " << muonP4_limited.eta() << ", phi = " << muonP4_limited.phi() << std::endl;
    std::cout << "other: En = " << otherP4_limited.E() << ", Pt = " << otherP4_limited.pt() << ", eta = " << otherP4_limited.eta() << ", phi = " << otherP4_limited.phi() << std::endl;
    std::cout << "Z: En = " << zP4.E() << ", Pt = " << zP4.pt() << ", eta = " << zP4.eta() << ", phi = " << zP4.phi() << std::endl;
  }

  reco::Candidate::LorentzVector sumPhotonP4(0.,0.,0.,0.);

  if ( mode_ == kPYTHIA ) {
    gen::Pythia6Service::InstanceWrapper pythia6InstanceGuard(pythia_);
    
    if ( !pythia_isInitialized_ ) {
      pythia_->setGeneralParams();
      pythia_->setCSAParams();
      pythia_->setSLHAParams();   
      gen::call_pygive("MSTU(12) = 12345"); 
      //gen::call_pygive("MSTU(10) = 1");     
      //gen::call_pygive("MSTU(10) = 2");   
      call_pyinit("NONE", "", "", 0.);
      pythia_isInitialized_ = true;
    }

    int zPythiaId = HepPID::translatePDTtoPythia(zPdgId);
    call_py1ent(1, zPythiaId, zP4.E(), zP4.theta(), zP4.phi());
    pyjets.k[0][0] = 11;
    pyjets.k[3][0] = 2;
    pyjets.k[4][0] = 3;
    int muonPythiaId = HepPID::translatePDTtoPythia(muonPdgId);
    call_py1ent(2, muonPythiaId, muonP4.E(), muonP4.theta(), muonP4.phi());
    //pyjets.p[4][1] = gen::pymass_(muonPythiaId);
    pyjets.k[2][1] = 1;
    int otherPythiaId = HepPID::translatePDTtoPythia(otherPdgId);
    call_py1ent(3, otherPythiaId, otherP4.E(), otherP4.theta(), otherP4.phi());
    //pyjets.p[4][2] = gen::pymass_(otherPythiaId);
    pyjets.k[2][2] = 1;
        
    std::cout << "PYJETS (beforeFSR):" << std::endl;
    printPYJETS();

    call_pyexec();

    std::cout << "PYJETS (afterFSR):" << std::endl;
    printPYJETS();

    // find lowest energy muon
    // (= muon after FSR) 
    reco::Candidate::LorentzVector muonP4afterFSR = muonP4;
    for ( int iParticle = 0; iParticle < pyjets.n; ++iParticle ) {
      if ( pyjets.k[1][iParticle] == HepPID::translatePDTtoPythia(muonPdgId) && pyjets.p[3][iParticle] < muonP4afterFSR.E() ) {
	muonP4afterFSR.SetPxPyPzE(
	  pyjets.p[0][iParticle],
	  pyjets.p[1][iParticle],
	  pyjets.p[2][iParticle],
	  pyjets.p[3][iParticle]);
      }	 
    }
    sumPhotonP4 = muonP4 - muonP4afterFSR;
  } else if ( mode_ == kPHOTOS ) {    

    if ( !photos_isInitialized_ ) {
      photos_->init();
      photos_isInitialized_ = true;
    }
    
    HepMC::GenEvent* genEvt_beforeFSR = new HepMC::GenEvent();
    HepMC::GenVertex* genVtx = new HepMC::GenVertex(HepMC::FourVector(0.,0.,0.,0.));
    HepMC::GenParticle* genZ = new HepMC::GenParticle((HepMC::FourVector)zP4, zPdgId, 2, HepMC::Flow(), HepMC::Polarization(0,0));
    genVtx->add_particle_in(genZ);
    HepMC::GenParticle* genMuon = new HepMC::GenParticle((HepMC::FourVector)muonP4_limited, muonPdgId, 1, HepMC::Flow(), HepMC::Polarization(0,0));
    genVtx->add_particle_out(genMuon);
    HepMC::GenParticle* genOther = new HepMC::GenParticle((HepMC::FourVector)otherP4_limited, otherPdgId, 1, HepMC::Flow(), HepMC::Polarization(0,0));
    genVtx->add_particle_out(genOther);
    genEvt_beforeFSR->add_vertex(genVtx);

    if ( verbosity_ ) {
      std::cout << "genEvt (beforeFSR):" << std::endl;
      genEvt_beforeFSR->print(std::cout);
    }
      
    HepMC::IO_HEPEVT conv;
    conv.write_event(genEvt_beforeFSR); 
    HepMC::GenEvent* genEvt_afterFSR = photos_->apply(genEvt_beforeFSR);

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
    sumPhotonP4 = muonP4 - muonP4afterFSR;

    delete genEvt_beforeFSR;
    if ( genEvt_afterFSR != genEvt_beforeFSR ) delete genEvt_afterFSR;
  } else assert(0);
  
  if ( verbosity_ ) {
    std::cout << "sumPhotons: En = " << sumPhotonP4.E() << ", Pt = " << sumPhotonP4.pt() << ", eta = " << sumPhotonP4.eta() << ", phi = " << sumPhotonP4.phi() << std::endl;
  }
  
  return sumPhotonP4;
}
