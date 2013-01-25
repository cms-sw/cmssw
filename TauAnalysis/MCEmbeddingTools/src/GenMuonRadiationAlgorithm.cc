#include "TauAnalysis/MCEmbeddingTools/interface/GenMuonRadiationAlgorithm.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "GeneratorInterface/ExternalDecays/interface/DecayRandomEngine.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "HepMC/IO_HEPEVT.h"
#include "HepMC/PythiaWrapper6_4.h"
#include "TauAnalysis/MCEmbeddingTools/interface/extraPythia.h"                // needed for call_pyexec
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Declarations.h" // needed for call_pyhepc
#include "HepPID/ParticleIDTranslations.hh"

#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"
#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <Math/VectorUtil.h>

const double protonMass = 0.938272;

bool GenMuonRadiationAlgorithm::photos_isInitialized_ = false;
bool GenMuonRadiationAlgorithm::pythia_isInitialized_ = false;

class myPythia6ServiceWithCallback : public gen::Pythia6Service 
{
 public:
  myPythia6ServiceWithCallback(const edm::ParameterSet& cfg) 
    : gen::Pythia6Service(cfg),
      beamEnergy_(cfg.getParameter<double>("beamEnergy")),
      genEvt_(0)
  {}
  ~myPythia6ServiceWithCallback() {}

  void setEvent(const HepMC::GenEvent* genEvt) 
  {  
    genEvt_ = genEvt;
  }
 
 private:
  void upInit() 
  { 
    // initialize LEs Houches accord common block (see hep-ph/0109068)
    // needed to run PYTHIA in parton shower mode
    std::cout << "<upInit>:" << std::endl;
    lhef::HEPRUP heprup;
    heprup.IDBMUP.first  = 2212;
    heprup.IDBMUP.second = 2212;
    heprup.EBMUP.first   = beamEnergy_;
    heprup.EBMUP.second  = beamEnergy_;
    heprup.PDFGUP.first  = 0;
    heprup.PDFGUP.second = 0;
    heprup.PDFSUP.first  = 10042;
    heprup.PDFSUP.second = 10042;
    heprup.IDWTUP        = 3;
    heprup.NPRUP         = 1;
    heprup.XSECUP.push_back(1.e+3);
    heprup.XERRUP.push_back(0.);
    heprup.XMAXUP.push_back(1.);
    heprup.LPRUP.push_back(1);
    lhef::CommonBlocks::fillHEPRUP(&heprup);   
  }
  void upEvnt() 
  {     
    std::cout << "<upEvnt>:" << std::endl;
    int numParticles = 0;
    for ( HepMC::GenEvent::particle_const_iterator genParticle = genEvt_->particles_begin();
	  genParticle != genEvt_->particles_end(); ++genParticle ) {
      ++numParticles;
    }
    std::cout << " numParticles = " << numParticles << std::endl;
    pyjets.n = numParticles;
    std::map<int, int> barcodeToIdxMap;
    int genParticleIdx = 0;
    for ( HepMC::GenEvent::particle_const_iterator genParticle = genEvt_->particles_begin();
	  genParticle != genEvt_->particles_end(); ++genParticle ) {
      barcodeToIdxMap[(*genParticle)->barcode()] = genParticleIdx + 1; // CV: conversion between C++ and Fortran array index conventions
      ++genParticleIdx;
    }
    genParticleIdx = 0;
    for ( HepMC::GenEvent::particle_const_iterator genParticle = genEvt_->particles_begin();
	  genParticle != genEvt_->particles_end(); ++genParticle ) {
      pyjets.k[0][genParticleIdx] = (*genParticle)->status();
      pyjets.k[1][genParticleIdx] = HepPID::translatePDTtoPythia((*genParticle)->pdg_id());
      if ( (*genParticle)->production_vertex() ) {
	int firstMother = 0;
	bool firstMother_initialized = false;
	int lastMother = 0;
	bool lastMother_initialized = false;
	for ( HepMC::GenVertex::particles_in_const_iterator genMother = (*genParticle)->production_vertex()->particles_in_const_begin();
	      genMother != (*genParticle)->production_vertex()->particles_in_const_end(); ++genMother ) {
	  int genMotherBarcode = (*genMother)->barcode();
          assert(barcodeToIdxMap.find(genMotherBarcode) != barcodeToIdxMap.end());
	  int genMotherIdx = barcodeToIdxMap[genMotherBarcode];
	  if ( genMotherIdx < firstMother || !firstMother_initialized ) {
	    firstMother = genMotherIdx;
	    firstMother_initialized = true;
	  }
	  if ( genMotherIdx > lastMother || !lastMother_initialized ) {
	    lastMother = genMotherIdx;
	    lastMother_initialized = true;
	  }
	}
	pyjets.k[2][genParticleIdx] = firstMother;
      } else {
	pyjets.k[2][genParticleIdx] = 0;
      }
      if ( (*genParticle)->end_vertex() ) {
	int firstDaughter = 0;
	bool firstDaughter_initialized = false;
	int lastDaughter = 0;
	bool lastDaughter_initialized = false;
	for ( HepMC::GenVertex::particles_in_const_iterator genDaughter = (*genParticle)->end_vertex()->particles_out_const_begin();
	      genDaughter != (*genParticle)->end_vertex()->particles_out_const_end(); ++genDaughter ) {
	  int genDaughterBarcode = (*genDaughter)->barcode();
          assert(barcodeToIdxMap.find(genDaughterBarcode) != barcodeToIdxMap.end());
	  int genDaughterIdx = barcodeToIdxMap[genDaughterBarcode];
	  if ( genDaughterIdx < firstDaughter || !firstDaughter_initialized ) {
	    firstDaughter = genDaughterIdx;
	    firstDaughter_initialized = true;
	  }
	  if ( genDaughterIdx > lastDaughter || !lastDaughter_initialized ) {
	    lastDaughter = genDaughterIdx;
	    lastDaughter_initialized = true;
	  }
	}
	pyjets.k[3][genParticleIdx] = firstDaughter;
	pyjets.k[4][genParticleIdx] = lastDaughter;
      } else {
	pyjets.k[3][genParticleIdx] = 0;
	pyjets.k[4][genParticleIdx] = 0;
      }
      pyjets.p[0][genParticleIdx] = (*genParticle)->momentum().px();
      pyjets.p[1][genParticleIdx] = (*genParticle)->momentum().py();
      pyjets.p[2][genParticleIdx] = (*genParticle)->momentum().pz();
      pyjets.p[3][genParticleIdx] = (*genParticle)->momentum().e();
      pyjets.p[4][genParticleIdx] = (*genParticle)->momentum().m();
      if ( (*genParticle)->production_vertex() ) {
	pyjets.v[0][genParticleIdx] = (*genParticle)->production_vertex()->position().x();
	pyjets.v[1][genParticleIdx] = (*genParticle)->production_vertex()->position().y();
	pyjets.v[2][genParticleIdx] = (*genParticle)->production_vertex()->position().z();
	pyjets.v[3][genParticleIdx] = (*genParticle)->production_vertex()->position().t();
      } else {
	for ( int i = 0; i < 4; ++i ) {
	  pyjets.v[i][genParticleIdx] = 0.;
	}
      }
      pyjets.v[4][genParticleIdx] = 0.;
      std::cout << "adding genParticle #" << genParticleIdx 
		<< " (pythiaId = " << pyjets.k[1][genParticleIdx] << ", status = " << pyjets.k[0][genParticleIdx] << "):" 
		<< " En = " << pyjets.p[3][genParticleIdx] << "," 
		<< " Px = " << pyjets.p[0][genParticleIdx] << "," 
		<< " Py = " << pyjets.p[1][genParticleIdx] << "," 
		<< " Pz = " << pyjets.p[2][genParticleIdx] << " (mass = " << pyjets.p[4][genParticleIdx] << ")" << std::endl;
      ++genParticleIdx;
    }
  }
  bool upVeto() 
  { 
    std::cout << "<upVeto>:" << std::endl;
    return false; 
  }

  double beamEnergy_;

  const HepMC::GenEvent* genEvt_;
};

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

  if ( mode_ == kPYTHIA ) pythia_ = new myPythia6ServiceWithCallback(cfg);
  if ( mode_ == kPHOTOS ) photos_ = new gen::PhotosInterface(cfg.getParameter<edm::ParameterSet>("PhotosOptions"));

  maxTrials_ = 10;

  verbosity_ = ( cfg.exists("verbosity") ) ? 
    cfg.getParameter<int>("verbosity") : 0;
}

GenMuonRadiationAlgorithm::~GenMuonRadiationAlgorithm()
{
  delete photos_;
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

  reco::Candidate::LorentzVector boostToLab(const reco::Candidate::LorentzVector& rfSystem,
					    const reco::Candidate::LorentzVector& p4ToBoost) 
  {
    reco::Candidate::Vector boost = rfSystem.BoostToCM();
    return ROOT::Math::VectorUtil::boost(p4ToBoost, -boost);
  }

  void printPythiaParameters()
  {
    call_pylist(12);
    call_pylist(13);
    std::cout << "\n PYPARS \n" << std::endl;
    std::cout << std::setw(5) << std::fixed << "I"
	      << std::setw(10) << std::fixed << "MSTP(I)"
	      << std::setw(16) << std::fixed << "PARP(I)"
	      << std::setw(10) << std::fixed << "MSTI(I)"
	      << std::setw(16) << std::fixed << "PARI(I)" << std::endl;
    for ( unsigned int ind=0; ind < 200; ind++ ) {
      std::cout << std::setw(5) << std::fixed << ind+1
		<< std::setw(10) << std::fixed << pypars.mstp[ind]
		<< std::setw(16) << std::fixed << pypars.parp[ind]
		<< std::setw(10) << std::fixed << pypars.msti[ind]
		<< std::setw(16) << std::fixed << pypars.pari[ind] << std::endl;
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

  errorFlag = 0;

  HepMC::GenEvent* genEvt_beforeFSR = new HepMC::GenEvent();

  HepMC::GenVertex* genVtx_in = new HepMC::GenVertex(HepMC::FourVector(0.,0.,0.,0.));
  double protonEn = beamEnergy_;
  double protonPz = sqrt(square(protonEn) - square(protonMass));
  genVtx_in->add_particle_in(new HepMC::GenParticle(HepMC::FourVector(0., 0., +protonPz, protonEn), 2212, 3));
  genVtx_in->add_particle_in(new HepMC::GenParticle(HepMC::FourVector(0., 0., -protonPz, protonEn), 2212, 3));
  genEvt_beforeFSR->add_vertex(genVtx_in);

  HepMC::GenVertex* genVtx_out = new HepMC::GenVertex(HepMC::FourVector(0.,0.,0.,0.));
  
  reco::Candidate::LorentzVector muonP4_limited = getP4_limited(muonP4, muonP4.mass());  
  reco::Candidate::LorentzVector otherP4_limited = getP4_limited(otherP4, otherP4.mass());
  reco::Candidate::LorentzVector sumP4 = muonP4_limited + otherP4_limited;
  if ( verbosity_ ) {
    std::cout << "muon(limited): En = " << muonP4_limited.E() << ", Pt = " << muonP4_limited.pt() << ", eta = " << muonP4_limited.eta() << ", phi = " << muonP4_limited.phi() << std::endl;
    std::cout << "other: En = " << otherP4_limited.E() << ", Pt = " << otherP4_limited.pt() << ", eta = " << otherP4_limited.eta() << ", phi = " << otherP4_limited.phi() << std::endl;
    std::cout << "Z: En = " << sumP4.E() << ", Pt = " << sumP4.pt() << ", eta = " << sumP4.eta() << ", phi = " << sumP4.phi() << std::endl;
  }
  // CV: use pdgId code for Z
  int zPdgId = 23;
  HepMC::GenParticle* genZ = new HepMC::GenParticle((HepMC::FourVector)sumP4, zPdgId, 2, HepMC::Flow(), HepMC::Polarization(0,0));
  genVtx_in->add_particle_out(genZ);
  reco::Candidate::LorentzVector ppP4_scattered(-sumP4.px(), -sumP4.py(), -sumP4.pz(), 2.*beamEnergy_ - sumP4.E());
  double protonEn_rf = 0.5*ppP4_scattered.mass();
  double protonP_rf  = sqrt(square(protonEn_rf) - square(protonMass));
  reco::Candidate::LorentzVector proton1P4_scattered_rf(0., 0., +protonP_rf, protonEn_rf);
  reco::Candidate::LorentzVector proton2P4_scattered_rf(0., 0., -protonP_rf, protonEn_rf);
  reco::Candidate::LorentzVector proton1P4_scattered = boostToLab(ppP4_scattered, proton1P4_scattered_rf);
  reco::Candidate::LorentzVector proton2P4_scattered = boostToLab(ppP4_scattered, proton2P4_scattered_rf);
  genVtx_in->add_particle_out(new HepMC::GenParticle(HepMC::FourVector(proton1P4_scattered.px(), proton1P4_scattered.py(), proton1P4_scattered.pz(), proton1P4_scattered.E()), 2212, 1));
  genVtx_in->add_particle_out(new HepMC::GenParticle(HepMC::FourVector(proton2P4_scattered.px(), proton2P4_scattered.py(), proton2P4_scattered.pz(), proton2P4_scattered.E()), 2212, 1));
  genVtx_out->add_particle_in(genZ);

  int muonPdgId = -13*muonCharge;
  HepMC::GenParticle* genMuon = new HepMC::GenParticle((HepMC::FourVector)muonP4_limited, muonPdgId, 1, HepMC::Flow(), HepMC::Polarization(0,0));
  genVtx_out->add_particle_out(genMuon);

  int otherPdgId = +13*muonCharge;
  HepMC::GenParticle* genOther = new HepMC::GenParticle((HepMC::FourVector)otherP4_limited, otherPdgId, 1, HepMC::Flow(), HepMC::Polarization(0,0));
  genVtx_out->add_particle_out(genOther);

  genEvt_beforeFSR->add_vertex(genVtx_in);
  genEvt_beforeFSR->add_vertex(genVtx_out);

  repairBarcodes(genEvt_beforeFSR);

  if ( verbosity_ ) {
    std::cout << "genEvt (beforeFSR):" << std::endl;
    genEvt_beforeFSR->print(std::cout);
  }

  HepMC::GenEvent* genEvt_afterFSR = 0;
  
  if ( mode_ == kPYTHIA ) {
    gen::Pythia6Service::InstanceWrapper pythia6InstanceGuard(pythia_);

    if ( !pythia_isInitialized_ ) {
      pythia_->setGeneralParams();      
      //pythia_->setPYUPDAParams(false);
      call_pyinit("USER", "", "", 0.);
      //pythia_->setPYUPDAParams(true);
      //printPythiaParameters();
      pythia_isInitialized_ = true;
    }

    pythia_->setEvent(genEvt_beforeFSR);
    call_pylist(2);
    call_pyevnt();
    call_pylist(2);

    HepMC::IO_HEPEVT conv;
    call_pyhepc(1); 
    genEvt_afterFSR = conv.read_next_event();
  } else if ( mode_ == kPHOTOS ) {
    if ( !photos_isInitialized_ ) {
      photos_->init();
      photos_isInitialized_ = true;
    }
    
    HepMC::IO_HEPEVT conv;
    conv.write_event(genEvt_beforeFSR); 
    genEvt_afterFSR = photos_->apply(genEvt_beforeFSR);
  } else assert(0);

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
