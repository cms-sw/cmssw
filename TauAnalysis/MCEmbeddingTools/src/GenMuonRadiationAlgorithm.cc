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

#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"
#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

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

  void setEvent(const HepMC::GenEvent* genEvt) 
  {  
    genEvt_ = genEvt;
  }
 
 private:
  void upInit() 
  { 
    // initialize LEs Houches accord common block (see hep-ph/0109068)
    // needed to run PYTHIA in parton shower mode
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
    int numParticles = 0;
    for ( HepMC::GenEvent::particle_const_iterator genParticle = genEvt_->particles_begin();
	  genParticle != genEvt_->particles_end(); ++genParticle ) {
      ++numParticles;
    }
    lhef::HEPEUP hepeup;
    hepeup.NUP = numParticles;
    hepeup.IDPRUP = 1;
    hepeup.XWGTUP = 1.;
    hepeup.XPDWUP.first = 0.;
    hepeup.XPDWUP.second = 0.;
    hepeup.SCALUP = -1.;
    hepeup.AQEDUP = -1.;
    hepeup.AQCDUP = -1.;
    hepeup.resize();
    std::map<int, int> barcodeToIdxMap;
    int genParticleIdx = 0;
    for ( HepMC::GenEvent::particle_const_iterator genParticle = genEvt_->particles_begin();
	  genParticle != genEvt_->particles_end(); ++genParticle ) {
      barcodeToIdxMap[(*genParticle)->barcode()] = genParticleIdx + 1; // CV: conversion between C++ and Fortran array index conventions
      hepeup.IDUP[genParticleIdx] = (*genParticle)->pdg_id();
      if ( (*genParticle)->production_vertex() ) {
	hepeup.ISTUP[genParticleIdx] = (*genParticle)->status();
      } else {
	hepeup.ISTUP[genParticleIdx] = -1;
      }
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
	hepeup.MOTHUP[genParticleIdx].first = firstMother;
	hepeup.MOTHUP[genParticleIdx].second = lastMother;
      } else {
	hepeup.MOTHUP[genParticleIdx].first = 0;
	hepeup.MOTHUP[genParticleIdx].second = 0;
      }
      hepeup.ICOLUP[genParticleIdx].first = 0;
      hepeup.ICOLUP[genParticleIdx].second = 0;
      hepeup.PUP[genParticleIdx][0] = (*genParticle)->momentum().px();
      hepeup.PUP[genParticleIdx][1] = (*genParticle)->momentum().py();
      hepeup.PUP[genParticleIdx][2] = (*genParticle)->momentum().pz();
      hepeup.PUP[genParticleIdx][3] = (*genParticle)->momentum().e();
      hepeup.PUP[genParticleIdx][4] = (*genParticle)->momentum().m();
      if ( (*genParticle)->status() == 1 ) hepeup.VTIMUP[genParticleIdx] = 1.e+6;
      else hepeup.VTIMUP[genParticleIdx] = 0.;
      hepeup.SPINUP[genParticleIdx] = 9;
      ++genParticleIdx;
    }
    lhef::CommonBlocks::fillHEPEUP(&hepeup);
  }
  bool upVeto() 
  { 
    return true; 
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

  double cube(double x)
  {
    return x*x*x;
  }

  double fourth(double x)
  {
    return x*x*x*x;
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
}

reco::Candidate::LorentzVector GenMuonRadiationAlgorithm::compFSR(const reco::Candidate::LorentzVector& muonP4, int muonCharge,
								  const reco::Candidate::LorentzVector& otherP4)
{
  if ( verbosity_ ) {
    std::cout << "<GenMuonRadiationAlgorithm::compMuonFSR>:" << std::endl;
    std::cout << " muon: En = " << muonP4.E() << ", Pt = " << muonP4.pt() << ", eta = " << muonP4.eta() << ", phi = " << muonP4.phi() << ", charge = " << muonCharge << std::endl;
    std::cout << " other: En = " << otherP4.E() << ", Pt = " << otherP4.pt() << ", eta = " << otherP4.eta() << ", phi = " << otherP4.phi() << std::endl;
  }

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
  double protonPx_scattered = -0.5*sumP4.px();
  double protonPy_scattered = -0.5*sumP4.py();
  double term1 = -0.5*sumP4.pz();
  double term2 = 0.5*sqrt(square(4.*beamEnergy_ - 2.*sumP4.E()) - 4.*square(sumP4.pz()));
  double term3 = 64.*fourth(beamEnergy_) - 128.*cube(beamEnergy_)*sumP4.E() + 4.*fourth(sumP4.E());
  double term4 = square(beamEnergy_)*(96.*square(sumP4.E()) - 16.*square(sumP4.P()) - 64.*square(protonMass));
  double term5 = -square(beamEnergy_)*(4.*square(sumP4.pz()) + 16.*square(protonMass));
  double term6 = -beamEnergy_*sumP4.E()*(32.*square(sumP4.E()) - 16.*square(sumP4.pz()) - 64.*square(protonMass));
  std::cout << "term1 = " << term1 << std::endl;
  std::cout << "term2 = " << term2 << std::endl;
  std::cout << "term3 = " << term3 << std::endl;
  std::cout << "term4 = " << term4 << std::endl;
  std::cout << "term5 = " << term5 << std::endl;
  std::cout << "term6 = " << term6 << std::endl;  
  double proton1Pz_scattered = term1 + term2*sqrt(term3 + term4 + term5 + term6);
  double proton1En_scattered = sqrt(square(proton1Pz_scattered) + square(protonMass));
  double proton2Pz_scattered = term1 - term2*sqrt(term3 + term4 + term5 + term6);
  double proton2En_scattered = sqrt(square(proton2Pz_scattered) + square(protonMass));
  genVtx_in->add_particle_out(new HepMC::GenParticle(HepMC::FourVector(0., 0., proton1Pz_scattered, proton1En_scattered), 2212, 1));
  genVtx_in->add_particle_out(new HepMC::GenParticle(HepMC::FourVector(0., 0., proton2Pz_scattered, proton2En_scattered), 2212, 1));
  genVtx_out->add_particle_in(genZ);

  int muonPdgId = -13*muonCharge;
  HepMC::GenParticle* genMuon = new HepMC::GenParticle((HepMC::FourVector)muonP4_limited, muonPdgId, 1, HepMC::Flow(), HepMC::Polarization(0,0));
  genVtx_out->add_particle_out(genMuon);

  //int neutrinoPdgId = +14*muonCharge;
  int neutrinoPdgId = +13*muonCharge;
  HepMC::GenParticle* genNeutrino = new HepMC::GenParticle((HepMC::FourVector)otherP4_limited, neutrinoPdgId, 1, HepMC::Flow(), HepMC::Polarization(0,0));
  genVtx_out->add_particle_out(genNeutrino);

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
      call_pyinit("USER", "p", "p", 2.*beamEnergy_);	
      pythia_->setGeneralParams();
      //call_pyinit("USER", "p", "p", 2.*beamEnergy_);
      pythia_isInitialized_ = true;
    }

    bool isValid = false;
    int numTries = 0;
    do {
      pythia_->setEvent(genEvt_beforeFSR);
      call_pyevnt();

      HepMC::IO_HEPEVT conv;
      call_pyhepc(1); 
      genEvt_afterFSR = conv.read_next_event();
      int numVertices = 0;
      for ( HepMC::GenEvent::vertex_iterator genVtx = genEvt_afterFSR->vertices_begin();
	    genVtx != genEvt_afterFSR->vertices_end(); ++genVtx ) {
	++numVertices;
      }
      if ( numVertices <= 3 ) isValid = true;
      else delete genEvt_afterFSR;
      ++numTries;
    } while ( !isValid && numTries < 1000 );
    if ( !isValid ) {
      edm::LogError ("GenMuonRadiationAlgorithm::compFSR") 
	<< "Failed to simulate muon -> muon + photon radiation using PYTHIA !!" << std::endl;
      std::cout << "genEvt:" << std::endl;
      genEvt_beforeFSR->print(std::cout);
      genEvt_afterFSR = genEvt_beforeFSR;
    }
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
