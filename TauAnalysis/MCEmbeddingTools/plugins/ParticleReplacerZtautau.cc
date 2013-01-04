#include "TauAnalysis/MCEmbeddingTools/plugins/ParticleReplacerZtautau.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/ExternalDecays/interface/DecayRandomEngine.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "HepPDT/ParticleDataTable.hh"

#include "HepMC/IO_HEPEVT.h"

#include "TauAnalysis/MCEmbeddingTools/interface/extraPythia.h"                // needed for call_pyexec
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Declarations.h" // needed for call_pyhepc

#include "DataFormats/Math/interface/deltaR.h"

#include <Math/VectorUtil.h>
#include <TMath.h>
#include <TVector3.h>

#include <stack>
#include <queue>

const double tauMass           = 1.77690;
const double muonMass          = 0.105658369;
const double electronMass      = 0.00051099893; // GeV

const double nomMassW          = 80.398;
const double breitWignerWidthW = 2.141;
const double nomMassZ          = 91.1876;
const double breitWignerWidthZ = 2.4952;

bool ParticleReplacerZtautau::tauola_isInitialized_ = false;

ParticleReplacerZtautau::ParticleReplacerZtautau(const edm::ParameterSet& cfg)
  : ParticleReplacerBase(cfg),
    generatorMode_(cfg.getParameter<std::string>("generatorMode")),
    beamEnergy_(cfg.getParameter<double>("beamEnergy")),
    tauola_(cfg.getParameter<edm::ParameterSet>("TauolaOptions")),
    pythia_(cfg)
{
  maxNumberOfAttempts_ = ( cfg.exists("maxNumberOfAttempts") ) ?
    cfg.getParameter<int>("maxNumberOfAttempts") : 1000;

  // transformationMode =
  //  0 - no transformation
  //  1 - mumu -> tautau
  //  2 - mumu -> ee
  //  3 - mumu -> taunu
  //  4 - munu -> taunu
  transformationMode_ = ( cfg.exists("transformationMode") ) ?
    cfg.getParameter<int>("transformationMode") : 1;
  if ( verbosity_ ) {
    edm::LogInfo("Replacer") << "generatorMode = " << generatorMode_ << "\n";
    edm::LogInfo("Replacer") << "transformationMode = " << transformationMode_ << "\n";
  }

  motherParticleID_ = ( cfg.exists("motherParticleID") ) ?
    cfg.getParameter<int>("motherParticleID") : 23;

  // require generator level visible tau decay products to exceed given transverse momentum.
  // The purpose of this flag is make maximal use of the available Zmumu events statistics
  // by not generating tau-paisr which will fail the visible Pt cuts applied on reconstruction level in physics analyses.
  //
  // NOTE: the thresholds specified by configuration parameter 'minVisibleTransverseMomentum' need to be a few GeV lower
  //       than the cuts applied on reconstruction level in physics analyses,
  //       to account for resolution effects 
  //
  if ( cfg.exists("minVisibleTransverseMomentum") ) {
    std::string minVisibleTransverseMomentumLine = cfg.getParameter<std::string>("minVisibleTransverseMomentum");
    const char* startptr = minVisibleTransverseMomentumLine.c_str();
    char* endptr;
    double d = strtod(startptr, &endptr);
    if ( *endptr == '\0' && endptr != startptr ) {
      // fallback for backwards compatibility: 
      // if it's a single number then use this as a threshold for both particles
      MinVisPtCut cutLeg1;
      cutLeg1.type_ = MinVisPtCut::kTAU;
      cutLeg1.threshold_ = d;
      cutLeg1.index_ = 0; 
      MinVisPtCut cutLeg2;
      cutLeg2.type_ = MinVisPtCut::kTAU;
      cutLeg2.threshold_ = d;
      cutLeg2.index_ = 1;
      MinVisPtCutCombination minVisPtCut;
      minVisPtCut.cut_string_ = minVisibleTransverseMomentumLine;
      minVisPtCut.cuts_.push_back(cutLeg1);
      minVisPtCut.cuts_.push_back(cutLeg2);
      minVisPtCuts_.push_back(minVisPtCut);
    } else {
      // string has new format: 
      // parse the minVisibleTransverseMomentum string
      for ( std::string::size_type prev = 0, pos = 0; prev < minVisibleTransverseMomentumLine.length(); prev = pos + 1) {
	pos = minVisibleTransverseMomentumLine.find(';', prev);
	if ( pos == std::string::npos ) pos = minVisibleTransverseMomentumLine.length();
	
	std::string sub = minVisibleTransverseMomentumLine.substr(prev, pos - prev);
	MinVisPtCutCombination minVisPtCut;
	minVisPtCut.cut_string_ = minVisibleTransverseMomentumLine;
	const char* sub_c = sub.c_str();
	while (*sub_c != '\0' ) {
	  const char* sep = std::strchr(sub_c, '_');
	  if ( sep == NULL ) throw cms::Exception("Configuration") 
	    << "Minimum transverse parameter string must contain an underscore to separate type from Pt threshold !!\n";
	  std::string type(sub_c, sep);
	  
	  MinVisPtCut cut;
	  if      ( type == "elec1" ) { cut.type_ = MinVisPtCut::kELEC; cut.index_ = 0; }
	  else if ( type == "mu1"   ) { cut.type_ = MinVisPtCut::kMU;   cut.index_ = 0; }
	  else if ( type == "had1"  ) { cut.type_ = MinVisPtCut::kHAD;  cut.index_ = 0; }
	  else if ( type == "tau1"  ) { cut.type_ = MinVisPtCut::kTAU;  cut.index_ = 0; }
	  else if ( type == "elec2" ) { cut.type_ = MinVisPtCut::kELEC; cut.index_ = 1; }
	  else if ( type == "mu2"   ) { cut.type_ = MinVisPtCut::kMU;   cut.index_ = 1; }
	  else if ( type == "had2"  ) { cut.type_ = MinVisPtCut::kHAD;  cut.index_ = 1; }
	  else if ( type == "tau2"  ) { cut.type_ = MinVisPtCut::kTAU;  cut.index_ = 1; }
	  else throw cms::Exception("Configuration") 
	    << "'" << type << "' is not a valid type. Allowed values are { elec1, mu1, had1, tau1, elec2, mu2, had2, tau2 } !!\n";
	  
	  char* endptr;
	  cut.threshold_ = strtod(sep + 1, &endptr);
	  if ( endptr == sep + 1 ) throw cms::Exception("Configuration") 
	    << "No Pt threshold given !!\n";
	  
	  std::cout << "Adding vis. Pt cut: index = " << cut.index_ << ", type = " << cut.type_ << ", threshold = " << cut.threshold_ << std::endl;
	  minVisPtCut.cuts_.push_back(cut);
	  sub_c = endptr;
	}
	minVisPtCuts_.push_back(minVisPtCut);
      }
    }
  }

  rfRotationAngle_ = cfg.getParameter<double>("rfRotationAngle")*TMath::Pi()/180.;

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( !rng.isAvailable() ) 
    throw cms::Exception("Configuration")
      << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
      << "which appears to be absent. Please add that service to your configuration\n"
      << "or remove the modules that require it.\n";

  // this is a global variable defined in GeneratorInterface/ExternalDecays/src/ExternalDecayDriver.cc
  decayRandomEngine = &rng->getEngine();
}

void ParticleReplacerZtautau::beginJob() 
{
  gen::Pythia6Service::InstanceWrapper pythia6InstanceGuard(&pythia_);
  pythia_.setGeneralParams();
}

namespace
{
  double square(double x)
  {
    return x*x;
  }

  bool matchesGenParticle(const HepMC::GenParticle* genParticle1, const HepMC::GenParticle* genParticle2)
  {
    if ( genParticle1->pdg_id() == genParticle2->pdg_id() && 
	 TMath::Abs(genParticle1->momentum().e() - genParticle2->momentum().e()) < (1.e-3*(genParticle1->momentum().e() + genParticle2->momentum().e())) &&
	 reco::deltaR(genParticle1->momentum(), genParticle2->momentum()) < 1.e-3 )
      return true;
    else
      return false;
  }

  bool matchesGenVertex(const HepMC::GenVertex* genVertex1, const HepMC::GenVertex* genVertex2, bool checkIncomingParticles, bool checkOutgoingParticles)
  {
    // require that vertex positions match
    if ( !(TMath::Abs(genVertex1->position().x() - genVertex2->position().x()) < 1.e-3 &&
	   TMath::Abs(genVertex1->position().y() - genVertex2->position().y()) < 1.e-3 &&
	   TMath::Abs(genVertex1->position().z() - genVertex2->position().z()) < 1.e-3) ) return false;

    // require that "incoming" particles match
    if ( checkIncomingParticles ) {
      for ( HepMC::GenVertex::particles_in_const_iterator genParticle1 = genVertex1->particles_in_const_begin();
	    genParticle1 != genVertex1->particles_in_const_end(); ++genParticle1 ) {
	bool isMatched = false;
	for ( HepMC::GenVertex::particles_in_const_iterator genParticle2 = genVertex2->particles_in_const_begin();
	      genParticle2 != genVertex2->particles_in_const_end() && !isMatched; ++genParticle2 ) {
	  isMatched |= matchesGenParticle(*genParticle1, *genParticle2);
	}
	if ( !isMatched ) return false;
      }
    }

    // require that "outgoing" particles match
    if ( checkOutgoingParticles ) {
      for ( HepMC::GenVertex::particles_out_const_iterator genParticle1 = genVertex1->particles_out_const_begin();
	    genParticle1 != genVertex1->particles_out_const_end(); ++genParticle1 ) {
	bool isMatched = false;
	for ( HepMC::GenVertex::particles_out_const_iterator genParticle2 = genVertex2->particles_out_const_begin();
	      genParticle2 != genVertex2->particles_out_const_end() && !isMatched; ++genParticle2 ) {
	  isMatched |= matchesGenParticle(*genParticle1, *genParticle2);
	}
	if ( !isMatched ) return false;
      }
    }

    return true;
  }
}

std::auto_ptr<HepMC::GenEvent> ParticleReplacerZtautau::produce(const std::vector<reco::Particle>& muons, const reco::Vertex* evtVtx, const HepMC::GenEvent* genEvt)
{
  if ( evtVtx != 0 ) throw cms::Exception("Configuration") 
    << "ParticleReplacerZtautau does NOT support using primary vertex as the origin for taus !!\n";

//--- transform the muons to the desired gen. particles
  std::vector<reco::Particle> embedParticles;	
  if ( transformationMode_ <= 2 ) { // mumu -> ll (Z -> Z boson, l = { e, mu, tau })
    
    if ( muons.size() != 2 ) {
      edm::LogError ("Replacer") 
	<< "The decay mode Z->ll requires exactly two muons --> returning empty HepMC event !!" << std::endl;
      return std::auto_ptr<HepMC::GenEvent>(0);
    }

    switch ( transformationMode_ ) {
    case 0: // mumu -> mumu
      targetParticle1Mass_     = muonMass;
      targetParticle1AbsPdgID_ = 13;
      break;
    case 1: // mumu -> tautau     
      targetParticle1Mass_     = tauMass;
      targetParticle1AbsPdgID_ = 15;      
      break;
    case 2: // mumu -> ee   
      targetParticle1Mass_     = electronMass;
      targetParticle1AbsPdgID_ = 11;
      break;
    default:
      assert(0);
    }
    targetParticle2Mass_     = targetParticle1Mass_;
    targetParticle2AbsPdgID_ = targetParticle1AbsPdgID_;
    
    const reco::Particle& muon1 = muons.at(0);
    const reco::Particle& muon2 = muons.at(1);
    reco::Particle embedLepton1(muon1.charge(), muon1.p4(), muon1.vertex(), muon1.pdgId(), 0, true);
    reco::Particle embedLepton2(muon2.charge(), muon2.p4(), muon2.vertex(), muon2.pdgId(), 0, true);
    if ( targetParticle1AbsPdgID_ != 13 ) {
      transformMuMu2LepLep(&embedLepton1, &embedLepton2);
    }
    embedParticles.push_back(embedLepton1);
    embedParticles.push_back(embedLepton2);
  } else if ( transformationMode_ == 3 ) { // mumu -> taunu (Z -> W boson)
    
    if ( muons.size() != 2 ) {
      edm::LogError ("Replacer") 
	<< "The decay mode W->taunu requires exactly two muons --> returning empty HepMC event !!" << std::endl;
      return std::auto_ptr<HepMC::GenEvent>(0);
    }
    
    targetParticle1Mass_     = tauMass;
    targetParticle1AbsPdgID_ = 15;   
    targetParticle2Mass_     = targetParticle1Mass_;
    targetParticle2AbsPdgID_ = targetParticle1AbsPdgID_;

    const reco::Particle& muon1 = muons.at(0);
    const reco::Particle& muon2 = muons.at(1);
    reco::Particle embedTau(muon1.charge(), muon1.p4(), muon1.vertex(), muon1.pdgId(), 0, true);
    reco::Particle embedNu(muon2.charge(), muon2.p4(), muon2.vertex(), muon2.pdgId(), 0, true);
    transformMuMu2TauNu(&embedTau, &embedNu);
    embedParticles.push_back(embedTau);
    embedParticles.push_back(embedNu);   
  } else if ( transformationMode_ == 4 ) { // munu -> taunu (W -> W boson)
    
    if ( muons.size() != 2 ) {
      edm::LogError ("Replacer") 
	<< "The decay mode W->taunu requires exactly two gen. leptons --> returning empty HepMC event !!" << std::endl;
      return std::auto_ptr<HepMC::GenEvent>(0);
    }

    targetParticle1Mass_     = tauMass;
    targetParticle1AbsPdgID_ = 15;   
    targetParticle2Mass_     = 0.;
    targetParticle2AbsPdgID_ = 16; 
			
    const reco::Particle& muon = muons.at(0);
    double embedLeptonEn = sqrt(square(muon.px()) + square(muon.py()) + square(muon.pz()) + square(targetParticle1Mass_));
    reco::Candidate::LorentzVector embedTauP4(muon.px(), muon.py(), muon.pz(), embedLeptonEn);
    reco::Particle embedTau(muon.charge(), embedTauP4, muon.vertex(), targetParticle1AbsPdgID_*muon.pdgId()/std::abs(muon.pdgId()), 0, true);
    embedTau.setStatus(1);
    embedParticles.push_back(embedTau);
    const reco::Particle& nu = muons.at(1);
    reco::Particle embedNu(0, nu.p4(), nu.vertex(), -targetParticle2AbsPdgID_*muon.pdgId()/std::abs(muon.pdgId()), 0, true);
    embedNu.setStatus(1);
    embedParticles.push_back(embedNu);
  }
	
  if ( embedParticles.size() != 2 ){
    edm::LogError ("Replacer") 
      << "The creation of gen. particles failed somehow --> returning empty HepMC event !!" << std::endl;	
    return std::auto_ptr<HepMC::GenEvent>(0);
  }

  HepMC::GenEvent* genEvt_output = 0;
  
  HepMC::GenVertex* genVtx_output = new HepMC::GenVertex();
  
//--- prepare the output HepMC event
  if ( genEvt ) { // embed gen. leptons into existing HepMC event
    genEvt_output = new HepMC::GenEvent(*genEvt);
    
    for ( HepMC::GenEvent::vertex_iterator genVtx = genEvt_output->vertices_begin(); 
	  genVtx != genEvt_output->vertices_end(); ++genVtx ) {
      
      if ( (*genVtx)->particles_out_size() <= 0 || (*genVtx)->particles_in_size() <= 0 ) continue;
      
      bool foundMuon1 = false;
      bool foundMuon2 = false;
      bool foundZ     = false;
      for (  HepMC::GenVertex::particles_out_const_iterator genParticle = (*genVtx)->particles_out_const_begin(); 
	    genParticle != (*genVtx)->particles_out_const_end(); ++genParticle ) {
	if ( (*genParticle)->pdg_id() ==  13 ) foundMuon1 = true;
	if ( (*genParticle)->pdg_id() == -13 ) foundMuon2 = true;
	if ( (*genParticle)->pdg_id() ==  23 ) foundZ     = true;
      }
      
      int motherPdgId = (*(*genVtx)->particles_in_const_begin())->pdg_id();
      
      if ( ((*genVtx)->particles_out_size() ==  2 && 
	    (*genVtx)->particles_in_size()  >   0 &&
	    motherPdgId                     == 23 && 
	    foundMuon1                            &&
	    foundMuon2                            ) ||
	   ((*genVtx)->particles_out_size() >   2 &&
	    (*genVtx)->particles_in_size()  >   0 &&
	    motherPdgId                     == 23 && 
	    foundMuon1                            &&
	    foundMuon2                            &&
	    foundZ                                ) ) {
	genVtx_output = (*genVtx);
      }
    }
    
    cleanEvent(genEvt_output, genVtx_output);
    
    // prevent a decay of existing particles
    // (the decay of existing particles is a bug in the PythiaInterface which should be fixed in newer versions) <-- to be CHECKed (*)
    for (  HepMC::GenEvent::particle_iterator genParticle = genEvt_output->particles_begin();
	  genParticle != genEvt_output->particles_end(); ++genParticle ) {
      (*genParticle)->set_status(0);
    }
        
    for ( std::vector<reco::Particle>::const_iterator embedParticle = embedParticles.begin();
	  embedParticle != embedParticles.end(); ++embedParticle ) {
      genVtx_output->add_particle_out(new HepMC::GenParticle((HepMC::FourVector)embedParticle->p4(), embedParticle->pdgId(), 1, HepMC::Flow(), HepMC::Polarization(0,0)));
    }
  } else { // embed gen. leptons into new (empty) HepMC event
    reco::Particle::LorentzVector genMotherP4;
    double ppCollisionPosX = 0.;
    double ppCollisionPosY = 0.;
    double ppCollisionPosZ = 0.;
    int idx = 0;
    for ( std::vector<reco::Particle>::const_iterator embedParticle = embedParticles.begin();
	  embedParticle != embedParticles.end(); ++embedParticle ) {
      //std::cout << "embedParticle #" << idx << ": Pt = " << embedParticle->pt() << "," 
      //	  << " eta = " << embedParticle->eta() << ", phi = " << embedParticle->phi() << ", mass = " << embedParticle->mass() << std::endl;
      //std::cout << "(production vertex: x = " << embedParticle->vertex().x() << ", y = " << embedParticle->vertex().y() << ", z = " << embedParticle->vertex().z() << ")" << std::endl;
      genMotherP4 += embedParticle->p4();
      const reco::Particle::Point& embedParticleVertex = embedParticle->vertex();
      ppCollisionPosX += embedParticleVertex.x();
      ppCollisionPosY += embedParticleVertex.y();
      ppCollisionPosZ += embedParticleVertex.z();
      ++idx;
    }
    
    int numEmbedParticles = embedParticles.size();
    if ( numEmbedParticles > 0 ) {
      ppCollisionPosX /= numEmbedParticles;
      ppCollisionPosY /= numEmbedParticles;
      ppCollisionPosZ /= numEmbedParticles;
    }
    
    HepMC::GenVertex* ppCollisionVtx = new HepMC::GenVertex(HepMC::FourVector(ppCollisionPosX*10., ppCollisionPosY*10., ppCollisionPosZ*10., 0.)); // convert from cm to mm
    ppCollisionVtx->add_particle_in(new HepMC::GenParticle(HepMC::FourVector(0., 0.,  beamEnergy_, beamEnergy_), 2212, 3));
    ppCollisionVtx->add_particle_in(new HepMC::GenParticle(HepMC::FourVector(0., 0., -beamEnergy_, beamEnergy_), 2212, 3));

    HepMC::GenVertex* genMotherDecayVtx = new HepMC::GenVertex(HepMC::FourVector(ppCollisionPosX*10., ppCollisionPosY*10., ppCollisionPosZ*10., 0.)); // Z decays immediately
    HepMC::GenParticle* genMother = new HepMC::GenParticle((HepMC::FourVector)genMotherP4, motherParticleID_, (generatorMode_ == "Pythia" ? 3 : 2), HepMC::Flow(), HepMC::Polarization(0,0));
    if ( transformationMode_ == 3 ) {
      int chargedLepPdgId = embedParticles.begin()->pdgId(); // first daughter particle is charged lepton always
      int motherPdgId = -24*chargedLepPdgId/std::abs(chargedLepPdgId);
      genMother->set_pdg_id(motherPdgId);
    }

    ppCollisionVtx->add_particle_out(genMother);

    genMotherDecayVtx->add_particle_in(genMother);
    for ( std::vector<reco::Particle>::const_iterator embedParticle = embedParticles.begin();
	  embedParticle != embedParticles.end(); ++embedParticle ) {
      genMotherDecayVtx->add_particle_out(new HepMC::GenParticle((HepMC::FourVector)embedParticle->p4(), embedParticle->pdgId(), 1, HepMC::Flow(), HepMC::Polarization(0,0)));
    }

    genEvt_output = new HepMC::GenEvent();
    genEvt_output->add_vertex(ppCollisionVtx);
    genEvt_output->add_vertex(genMotherDecayVtx);

    repairBarcodes(genEvt_output);
  }
	
//--- compute probability to pass visible Pt cuts
  HepMC::GenEvent* passedEvt_output = 0;
  HepMC::GenEvent* tempEvt = 0;

  unsigned int numEvents_tried  = 0;
  unsigned int numEvents_passed = 0;
	
  int verbosity_backup = verbosity_;

  HepMC::IO_HEPEVT conv;
  for ( int iTrial = 0; iTrial < maxNumberOfAttempts_; ++iTrial ) {
    ++numEvents_tried;
    if ( generatorMode_ == "Pythia" )	{ // Pythia
      throw cms::Exception("Configuration") 
	<< "Pythia is currently not supported !!\n";
    } else if ( generatorMode_ == "Tauola" ) { // TAUOLA
      conv.write_event(genEvt_output);      
      tempEvt = tauola_.decay(genEvt_output);
    }
    
    bool passesVisPtCuts = testEvent(tempEvt);
    if ( passesVisPtCuts ) {
      if ( !passedEvt_output ) passedEvt_output = tempEvt; // store first HepMC event passing visible Pt cuts in output file
      else delete tempEvt;
      ++numEvents_passed;
      verbosity_ = 0;
    } else {
      delete tempEvt;
    }
  }

  tried_ = numEvents_tried;
  passed_ = numEvents_passed;
  verbosity_ = verbosity_backup;
  
  if ( !passedEvt_output ) {
    edm::LogError ("Replacer") 
      << "Failed to create an event which satisfies the visible Pt cuts !!" << std::endl;
    return std::auto_ptr<HepMC::GenEvent>(0);
  }

  // use PYTHIA to decay unstable hadrons (e.g. pi0 -> gamma gamma)
  //std::cout << "before pi0 -> gamma gamma decays:" << std::endl;
  //passedEvt_output->print(std::cout); 

  conv.write_event(passedEvt_output);

  gen::Pythia6Service::InstanceWrapper pythia6InstanceGuard(&pythia_);

  // convert hepevt -> pythia
  call_pyhepc(2); 

  // call PYTHIA 
  call_pyexec();
 
  // convert pythia -> hepevt
  call_pyhepc(1); 

  HepMC::GenEvent* passedEvt_pythia = conv.read_next_event();
  //std::cout << "PYTHIA output:" << std::endl;
  //passedEvt_pythia->print(std::cout);

  // CV: back and forth conversion between HepMC and PYTHIA causes
  //     pp -> Z vertex to get corrupted for some reason;
  //     do **not** take HepMC::GenEvent from PYTHIA, 
  //     but add decays of unstable particles to original HepMC::GenEvent
  for ( HepMC::GenEvent::vertex_const_iterator genVertex_pythia = passedEvt_pythia->vertices_begin();
	genVertex_pythia != passedEvt_pythia->vertices_end(); ++genVertex_pythia ) {
    int genVertex_barcode = (*genVertex_pythia)->barcode();

    bool isDecayVertex = ((*genVertex_pythia)->particles_in_size() >= 1 && (*genVertex_pythia)->particles_out_size() >= 2);
    for ( HepMC::GenEvent::vertex_const_iterator genVertex_output = passedEvt_output->vertices_begin();
	  genVertex_output != passedEvt_output->vertices_end(); ++genVertex_output ) {
      if ( matchesGenVertex(*genVertex_output, *genVertex_pythia, true, false) ) isDecayVertex = false;
    }
    if ( !isDecayVertex ) continue;

    // create new vertex
    //std::cout << "creating decay vertex: barcode = " << genVertex_barcode << std::endl;
    HepMC::GenVertex* genVertex_output = new HepMC::GenVertex((*genVertex_pythia)->position());

    // associate "incoming" particles to new vertex
    for ( HepMC::GenVertex::particles_in_const_iterator genParticle_pythia = (*genVertex_pythia)->particles_in_const_begin();
	  genParticle_pythia != (*genVertex_pythia)->particles_in_const_end(); ++genParticle_pythia ) {
      for ( HepMC::GenEvent::particle_iterator genParticle_output = passedEvt_output->particles_begin();
	    genParticle_output != passedEvt_output->particles_end(); ++genParticle_output ) {
	if ( matchesGenParticle(*genParticle_output, *genParticle_pythia) ) {
	  //std::cout << " adding 'incoming' particle: barcode = " << (*genParticle_output)->barcode() << std::endl;
	  genVertex_output->add_particle_in(*genParticle_output);
	}
      }
    }

    // create "outgoing" particles and associate them to new vertex
    for ( HepMC::GenVertex::particles_out_const_iterator genParticle_pythia = (*genVertex_pythia)->particles_out_const_begin();
	  genParticle_pythia != (*genVertex_pythia)->particles_out_const_end(); ++genParticle_pythia ) {
      HepMC::GenParticle* genParticle_output = new HepMC::GenParticle(
	(*genParticle_pythia)->momentum(), 
	(*genParticle_pythia)->pdg_id(), 
	(*genParticle_pythia)->status(), 
	(*genParticle_pythia)->flow(),  
	(*genParticle_pythia)->polarization());
      genParticle_output->suggest_barcode((*genParticle_pythia)->barcode());
      //std::cout << " adding 'outgoing' particle: barcode = " << genParticle_output->barcode() << std::endl;
      genVertex_output->add_particle_out(genParticle_output);
    }

    //std::cout << "adding decay vertex: barcode = " << genVertex_output->barcode() << std::endl;
    passedEvt_output->add_vertex(genVertex_output);
  }
  delete passedEvt_pythia;

  //std::cout << "after pi0 -> gamma gamma decays:" << std::endl;
  //passedEvt_output->print(std::cout);

  // undo the "hack" (*): 
  // recover the particle status codes
  if ( genEvt ) {
    for ( HepMC::GenEvent::particle_iterator genParticle = passedEvt_output->particles_begin();
	  genParticle != passedEvt_output->particles_end(); ++genParticle ) {
      if ( (*genParticle)->end_vertex() ) (*genParticle)->set_status(2);
      else (*genParticle)->set_status(1);
    }
  }
  
  std::auto_ptr<HepMC::GenEvent> passedEvt_output_ptr(passedEvt_output);    
  if ( verbosity_ ) {
    passedEvt_output_ptr->print(std::cout);
    std::cout << " numEvents: tried = " << numEvents_tried << ", passed = " << numEvents_passed << std::endl;
  }

  delete genVtx_output;
  delete genEvt_output;
  
  return passedEvt_output_ptr;
}

void ParticleReplacerZtautau::beginRun(edm::Run& run, const edm::EventSetup& es)
{
  if ( !tauola_isInitialized_ ) {
    std::cout << "<ParticleReplacerZtautau::beginRun>: Initializing TAUOLA interface." << std::endl;
    tauola_.init(es);
    tauola_isInitialized_ = true;
  }
}

void ParticleReplacerZtautau::endJob()
{
  tauola_.statistics();
}

bool ParticleReplacerZtautau::testEvent(HepMC::GenEvent* genEvt)
{
  //if ( verbosity_ ) std::cout << "<ParticleReplacerZtautau::testEvent>:" << std::endl;

  if ( minVisPtCuts_.empty() ) return true; // no visible Pt cuts applied

  std::vector<double> muonPts;
  std::vector<double> electronPts;
  std::vector<double> tauJetPts;
  std::vector<double> tauPts;

  int genParticleIdx = 0;
  for ( HepMC::GenEvent::particle_iterator genParticle = genEvt->particles_begin();
	genParticle != genEvt->particles_end(); ++genParticle ) {
    if ( abs((*genParticle)->pdg_id()) == 15 && (*genParticle)->end_vertex() ) {
      reco::Candidate::LorentzVector visP4;
      std::queue<const HepMC::GenParticle*> decayProducts;
      decayProducts.push(*genParticle);
      enum { kELEC, kMU, kHAD };
      int type = kHAD;
      int decayProductIdx = 0;
      while ( !decayProducts.empty() && decayProductIdx < 100 ) { // CV: protection against entering infinite loop in case of corrupt particle relations
	const HepMC::GenParticle* decayProduct = decayProducts.front();
	if ( verbosity_ ) {
	  std::cout << "decayProduct #" << (decayProductIdx + 1) << " (pdgId = " << decayProduct->pdg_id() << "):" 
		    << " Pt = " << decayProduct->momentum().perp() << ", eta = " << decayProduct->momentum().eta() << ", phi = " << decayProduct->momentum().phi() 
		    << std::endl;
	}
	decayProducts.pop();
	if ( !decayProduct->end_vertex() ) { // stable decay product
	  int absPdgId = abs(decayProduct->pdg_id());
	  if ( !(absPdgId == 12 || absPdgId == 14 || absPdgId == 16) ) visP4 += (reco::Candidate::LorentzVector)decayProduct->momentum();
          if ( absPdgId == 11 ) type = kELEC;
	  if ( absPdgId == 13 ) type = kMU;
	} else { // decay product decays further...
	  HepMC::GenVertex* decayVtx = decayProduct->end_vertex();
	  for ( HepMC::GenVertex::particles_out_const_iterator daughter = decayVtx->particles_out_const_begin();
		daughter != decayVtx->particles_out_const_end(); ++daughter ) {
	    decayProducts.push(*daughter);
	  }
	}
	++decayProductIdx;
      }

      double visPt = visP4.pt();
      tauPts.push_back(visPt);
      if      ( type == kMU   ) muonPts.push_back(visPt);
      else if ( type == kELEC ) electronPts.push_back(visPt);
      else if ( type == kHAD  ) tauJetPts.push_back(visPt);
      if ( verbosity_ ) {
        std::string type_string = "";
        if      ( type == kMU   ) type_string = "mu";
        else if ( type == kELEC ) type_string = "elec";
        else if ( type == kHAD  ) type_string = "had";
        std::cout << "visLeg #" << (genParticleIdx + 1) << " (type = " << type_string << "):" 
		  << " Pt = " << visP4.pt() << ", eta = " << visP4.eta() << ", phi = " << visP4.phi() 
		  << " (X = " << (visP4.energy()/(*genParticle)->momentum().e()) << ")" << std::endl;
      }
      ++genParticleIdx;
    }
  }

  std::sort(tauPts.begin(), tauPts.end(), std::greater<double>());
  std::sort(electronPts.begin(), electronPts.end(), std::greater<double>());
  std::sort(muonPts.begin(), muonPts.end(), std::greater<double>());
  std::sort(tauJetPts.begin(), tauJetPts.end(), std::greater<double>());
    
  // check if visible decay products pass Pt cuts
  //
  // NOTE: return value = True if leg1 > threshold[i] && leg2 > threshold[i] for **any** path i
  //      (e.g. (leg1Pt > 10 && leg2Pt > 20) || (leg1Pt > 20 && leg2Pt > 10), consistent with logic used by HLT)
  //
  for ( std::vector<MinVisPtCutCombination>::const_iterator minVisPtCut = minVisPtCuts_.begin();
	minVisPtCut != minVisPtCuts_.end(); ++minVisPtCut ) {
    //if ( verbosity_ ) minVisPtCut->print(std::cout);
    
    bool passesMinVisCut = true;
    
    for ( std::vector<MinVisPtCut>::const_iterator cut = minVisPtCut->cuts_.begin();
	  cut != minVisPtCut->cuts_.end(); ++cut ) {
      std::vector<double>* collection = 0;
      switch ( cut->type_ ) {
      case MinVisPtCut::kELEC:
	collection = &electronPts; 
	break;
      case MinVisPtCut::kMU: 
	collection = &muonPts; 
	break;
      case MinVisPtCut::kHAD: 
	collection = &tauJetPts; 
	break;
      case MinVisPtCut::kTAU: 
	collection = &tauPts; 
	break;
      }
      assert(collection);
      
      // j-th tau decay product fails visible Pt cut
      if ( cut->index_ >= collection->size() || (*collection)[cut->index_] < cut->threshold_ ) {
	passesMinVisCut = false;
	break;
      }
    }

    // all tau decay products satisfy visible Pt cuts for i-th path 
    //if ( verbosity_ ) std::cout << "passes vis. Pt cuts = " << passesMinVisCut << std::endl;
    if ( passesMinVisCut ) return true;
  }

  // visible Pt cuts failed for all paths
  return false;
}

void ParticleReplacerZtautau::cleanEvent(HepMC::GenEvent* genEvt, HepMC::GenVertex* genVtx)
{
  std::stack<HepMC::GenParticle*> genParticles_to_delete;
	
  std::stack<HepMC::GenVertex*> genVertices_to_process;
  std::stack<HepMC::GenVertex*> genVertices_to_delete;

  for ( HepMC::GenVertex::particles_out_const_iterator genParticle = genVtx->particles_out_const_begin();
	genParticle != genVtx->particles_out_const_end(); ++genParticle ) {
    genParticles_to_delete.push(*genParticle);
    if ( (*genParticle)->end_vertex() ) genVertices_to_process.push((*genParticle)->end_vertex());
  }

  while ( !genVertices_to_process.empty() ) {
    HepMC::GenVertex* tempVtx = genVertices_to_process.top();
    if ( tempVtx->particles_out_size() > 0 ) {
      for ( HepMC::GenVertex::particles_out_const_iterator genParticle = tempVtx->particles_out_const_begin();
	    genParticle != tempVtx->particles_out_const_end(); ++genParticle ) {
	if ( (*genParticle)->end_vertex() ) genVertices_to_process.push((*genParticle)->end_vertex());
      }
      delete tempVtx;
    }
    genVertices_to_delete.push(tempVtx);
    genVertices_to_process.pop();
  }
  	
  while ( !genVertices_to_delete.empty() ) {
    genEvt->remove_vertex(genVertices_to_delete.top());
    genVertices_to_delete.pop();
  }

  while ( !genParticles_to_delete.empty() ) {
    delete genVtx->remove_particle(genParticles_to_delete.top());
    genParticles_to_delete.pop();
  }

  repairBarcodes(genEvt);
}

void ParticleReplacerZtautau::repairBarcodes(HepMC::GenEvent* genEvt)
{
  int next_genVtx_barcode = 1;
  for ( HepMC::GenEvent::vertex_iterator genVtx = genEvt->vertices_begin();
	genVtx != genEvt->vertices_end(); ++genVtx ) {
    while ( !(*genVtx)->suggest_barcode(-1*next_genVtx_barcode) ) {
      ++next_genVtx_barcode;
    }
  }

  int next_genParticle_barcode = 1;
  for ( HepMC::GenEvent::particle_iterator genParticle = genEvt->particles_begin();
	genParticle != genEvt->particles_end(); ++genParticle ) {
    while ( !(*genParticle)->suggest_barcode(next_genParticle_barcode) ) {
      ++next_genParticle_barcode;
    }
  }
}

namespace
{
  reco::Particle::LorentzVector rotate(const reco::Particle::LorentzVector& p4, const reco::Particle::LorentzVector& axis, double angle)
  {
    TVector3 p3(p4.px(), p4.py(), p4.pz());
    p3.Rotate(angle, TVector3(axis.x(), axis.y(), axis.z()).Unit());
    reco::Particle::LorentzVector p4_rotated(p3.Px(), p3.Py(), p3.Pz(), p4.energy());
    assert(TMath::Abs(p3.Mag() - p4.P()) < (1.e-3*p4.P()));
    return p4_rotated;
  }

  void print(const std::string& label, const reco::Particle::LorentzVector& p4, const reco::Particle::LorentzVector* p4_ref = 0)
  {
    std::cout << label << ": En = " << p4.E() << ", Pt = " << p4.pt() << ", theta = " << p4.theta() << " (eta = " << p4.eta() << "), phi = " << p4.phi() << ", mass = " << p4.mass();
    if ( p4_ref ) {
      double angle = TMath::ACos((p4.px()*p4_ref->px() + p4.py()*p4_ref->py() + p4.pz()*p4_ref->pz())/(p4.P()*p4_ref->P()));
      std::cout << " (angle wrt. ref = " << angle << ")";
    }
    std::cout << std::endl;
  }
}

void ParticleReplacerZtautau::transformMuMu2LepLep(reco::Particle* muon1, reco::Particle* muon2)
{
//--- transform a muon pair into an electron/tau pair,
//    taking into account the difference between muon and electron/tau mass

  reco::Particle::LorentzVector muon1P4_lab = muon1->p4();
  reco::Particle::LorentzVector muon2P4_lab = muon2->p4();
  reco::Particle::LorentzVector zP4_lab = muon1P4_lab + muon2P4_lab;

  ROOT::Math::Boost boost_to_rf(zP4_lab.BoostToCM());
  ROOT::Math::Boost boost_to_lab(boost_to_rf.Inverse());

  reco::Particle::LorentzVector zP4_rf = boost_to_rf(zP4_lab);

  reco::Particle::LorentzVector muon1P4_rf = boost_to_rf(muon1P4_lab);
  reco::Particle::LorentzVector muon2P4_rf = boost_to_rf(muon2P4_lab);

  if ( verbosity_ ) {
    std::cout << "before rotation:" << std::endl;
    print("muon1(lab)", muon1P4_lab, &zP4_lab);
    print("muon2(lab)", muon2P4_lab, &zP4_lab);
    print("Z(lab)", zP4_lab);
    print("muon1(rf)", muon1P4_rf, &zP4_lab);
    print("muon2(rf)", muon2P4_rf, &zP4_lab);
    print("Z(rf)", zP4_rf);
  }

  if ( rfRotationAngle_ != 0. ) {
    double rfRotationAngle_value = rfRotationAngle_;
    if ( rfRotationAngle_ == -1. ) {
      double u = decayRandomEngine->flat();
      rfRotationAngle_value = 2.*TMath::Pi()*u;
    }
    
    muon1P4_rf = rotate(muon1P4_rf, zP4_lab, rfRotationAngle_value);
    muon2P4_rf = rotate(muon2P4_rf, zP4_lab, rfRotationAngle_value);
  }

  double muon1P_rf2 = square(muon1P4_rf.px()) + square(muon1P4_rf.py()) + square(muon1P4_rf.pz());
  double lep1Mass2 = square(targetParticle1Mass_);
  double lep1En_rf = 0.5*zP4_rf.E();
  double lep1P_rf2  = square(lep1En_rf) - lep1Mass2;
  if ( lep1P_rf2 < 0. ) lep1P_rf2 = 0.;
  float scaleFactor1 = sqrt(lep1P_rf2/muon1P_rf2);
  reco::Particle::LorentzVector lep1P4_rf = reco::Particle::LorentzVector(
    scaleFactor1*muon1P4_rf.px(), scaleFactor1*muon1P4_rf.py(), scaleFactor1*muon1P4_rf.pz(), lep1En_rf);

  double muon2P_rf2 = square(muon2P4_rf.px()) + square(muon2P4_rf.py()) + square(muon2P4_rf.pz());
  double lep2Mass2 = square(targetParticle2Mass_);
  double lep2En_rf = 0.5*zP4_rf.E();
  double lep2P_rf2  = square(lep2En_rf) - lep2Mass2;
  if ( lep2P_rf2 < 0. ) lep2P_rf2 = 0.;
  float scaleFactor2 = sqrt(lep2P_rf2/muon2P_rf2);
  reco::Particle::LorentzVector lep2P4_rf = reco::Particle::LorentzVector(
    scaleFactor2*muon2P4_rf.px(), scaleFactor2*muon2P4_rf.py(), scaleFactor2*muon2P4_rf.pz(), lep2En_rf);

  reco::Particle::LorentzVector lep1P4_lab = boost_to_lab(lep1P4_rf);
  reco::Particle::LorentzVector lep2P4_lab = boost_to_lab(lep2P4_rf);

  if ( verbosity_ ) {
    std::cout << "after rotation:" << std::endl;
    print("lep1(rf)", muon1P4_rf, &zP4_lab);
    print("lep2(rf)", muon2P4_rf, &zP4_lab);    
    reco::Particle::LorentzVector lep1p2_lab = lep1P4_lab + lep2P4_lab;
    print("lep1(lab)", lep1P4_lab, &zP4_lab);
    print("lep2(lab)", lep2P4_lab, &zP4_lab);    
    print("lep1+2(lab)", lep1p2_lab);
  }

  // perform additional checks:
  // the following tests guarantee a deviation of less than 0.1% 
  // for the following values of the original muons and the embedded electrons/taus in terms of:
  //  - invariant mass
  //  - transverse momentum
  if ( !(std::abs(zP4_lab.mass() - (lep1P4_lab + lep2P4_lab).mass())/zP4_lab.mass() < 1.e-3 &&
	 std::abs(zP4_lab.pt()   - (lep1P4_lab + lep2P4_lab).pt())/zP4_lab.pt()     < 1.e-3) ) 
    edm::LogError ("Replacer") 
      << "The kinematics of muons and embedded electrons/taus differ by more than 0.1%:" << std::endl 
      << " mass(muon1 + muon2) = " << zP4_lab.mass() << ", mass(lep1 + lep2) = " << (lep1P4_lab + lep2P4_lab).mass() << std::endl
      << " Pt(muon1 + muon2) = " << zP4_lab.pt() << ", Pt(lep1 + lep2) = " << (lep1P4_lab + lep2P4_lab).pt() << " --> please CHECK !!" << std::endl;

  muon1->setP4(lep1P4_lab);
  muon2->setP4(lep2P4_lab);

  muon1->setPdgId(targetParticle1AbsPdgID_*muon1->pdgId()/abs(muon1->pdgId())); 
  muon2->setPdgId(targetParticle2AbsPdgID_*muon2->pdgId()/abs(muon2->pdgId()));

  muon1->setStatus(1);
  muon2->setStatus(1);

  return;
}

void ParticleReplacerZtautau::transformMuMu2TauNu(reco::Particle* muon1, reco::Particle* muon2)
{
//--- transform a muon pair into tau + nu (replacing a Z by W boson)

  reco::Particle::LorentzVector muon1P4_lab = muon1->p4();
  reco::Particle::LorentzVector muon2P4_lab = muon2->p4();
  reco::Particle::LorentzVector zP4_lab = muon1P4_lab + muon2P4_lab;

  ROOT::Math::Boost boost_to_rf(zP4_lab.BoostToCM());
  ROOT::Math::Boost boost_to_lab(boost_to_rf.Inverse());

  reco::Particle::LorentzVector zP4_rf = boost_to_rf(zP4_lab);
  
  double wMass = (zP4_rf.mass() - nomMassZ)*(breitWignerWidthW/breitWignerWidthZ) + nomMassW;

  reco::Particle::LorentzVector muon1P4_rf = boost_to_rf(muon1P4_lab);
  reco::Particle::LorentzVector muon2P4_rf = boost_to_rf(muon2P4_lab);
  
  double muon1P_rf2 = square(muon1P4_rf.px()) + square(muon1P4_rf.py()) + square(muon1P4_rf.pz());
  double tauMass2 = square(targetParticle1Mass_);
  double tauEn_rf = 0.5*zP4_rf.E();
  double tauP_rf2  = square(tauEn_rf) - tauMass2;
  if ( tauP_rf2 < 0. ) tauP_rf2 = 0.;
  float scaleFactor1 = sqrt(tauP_rf2/muon1P_rf2)*(wMass/zP4_rf.mass());
  reco::Particle::LorentzVector tauP4_rf = reco::Particle::LorentzVector(
    scaleFactor1*muon1P4_rf.px(), scaleFactor1*muon1P4_rf.py(), scaleFactor1*muon1P4_rf.pz(), tauEn_rf);

  double muon2P_rf2 = square(muon2P4_rf.px()) + square(muon2P4_rf.py()) + square(muon2P4_rf.pz());
  double nuMass2 = square(targetParticle2Mass_);
  assert(nuMass2 < 1.e-4);
  double nuEn_rf = 0.5*zP4_rf.E();
  double nuP_rf2  = square(nuEn_rf) - nuMass2;
  if ( nuP_rf2 < 0. ) nuP_rf2 = 0.;
  float scaleFactor2 = sqrt(nuP_rf2/muon2P_rf2)*(wMass/zP4_rf.mass());
  reco::Particle::LorentzVector nuP4_rf = reco::Particle::LorentzVector(
    scaleFactor2*muon2P4_rf.px(), scaleFactor2*muon2P4_rf.py(), scaleFactor2*muon2P4_rf.pz(), nuEn_rf);
  
  reco::Particle::LorentzVector tauP4_lab = boost_to_lab(tauP4_rf);
  reco::Particle::LorentzVector nuP4_lab = boost_to_lab(nuP4_rf);

  // perform additional checks:
  // the following tests guarantee a deviation of less than 0.1% 
  // for the following values of the original muons and the embedded electrons/taus in terms of:
  //  - theta
  //  - phi
  if ( !(std::abs(zP4_lab.theta() - (tauP4_lab + nuP4_lab).theta())/zP4_lab.theta() < 1.e-3 &&
	 std::abs(zP4_lab.phi()   - (tauP4_lab + nuP4_lab).phi())/zP4_lab.phi()     < 1.e-3) ) 
    edm::LogError ("Replacer") 
      << "The kinematics of muons and embedded tau/neutrino differ by more than 0.1%:" << std::endl 
      << " mass(muon1 + muon2) = " << zP4_lab.mass() << ", mass(lep1 + lep2) = " << (tauP4_lab + nuP4_lab).mass() << std::endl
      << " Pt(muon1 + muon2) = " << zP4_lab.pt() << ", Pt(lep1 + lep2) = " << (tauP4_lab + nuP4_lab).pt() << " --> please CHECK !!" << std::endl;

  muon1->setP4(tauP4_lab);
  muon2->setP4(nuP4_lab);

  muon1->setPdgId(targetParticle1AbsPdgID_*muon1->pdgId()/abs(muon1->pdgId())); 
  muon2->setPdgId(targetParticle2AbsPdgID_*muon2->pdgId()/abs(muon2->pdgId()));

  muon1->setStatus(1);
  muon2->setStatus(1);

  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(ParticleReplacerPluginFactory, ParticleReplacerZtautau, "ParticleReplacerZtautau");
