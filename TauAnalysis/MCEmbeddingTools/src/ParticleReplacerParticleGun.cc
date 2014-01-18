#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerParticleGun.h"
#include "TauAnalysis/MCEmbeddingTools/interface/extraPythia.h"

#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HepMC/PythiaWrapper.h"
#include "HepMC/IO_HEPEVT.h"

using namespace edm;

namespace ParticleReplacerGunVar{
  CLHEP::HepRandomEngine* decayRandomEngine; // adding static var to replace missing value from ExternalDecays
}


ParticleReplacerParticleGun::ParticleReplacerParticleGun(const edm::ParameterSet& iConfig, bool verbose):
  ParticleReplacerBase(iConfig),
  pythia_(iConfig),
  particleOrigin_(iConfig.getParameter<std::string>("particleOrigin")),
  forceTauPolarization_(iConfig.getParameter<std::string>("forceTauPolarization")),
  forceTauDecay_(iConfig.getParameter<std::string>("forceTauDecay")),
  generatorMode_(iConfig.getParameter<std::string>("generatorMode")),
  gunParticle_(iConfig.getParameter<int>("gunParticle")),
  forceTauPlusHelicity_(iConfig.getParameter<int>("forceTauPlusHelicity")),
  forceTauMinusHelicity_(iConfig.getParameter<int>("forceTauMinusHelicity")),
  printout_(verbose) {
  tauola_ = (gen::TauolaInterfaceBase*)(TauolaFactory::get()->create("Tauolapp105", iConfig.getParameter<edm::ParameterSet>("ExternalDecays").getParameter<edm::ParameterSet>("Tauola"))); 
  //srand(time(NULL)); // Should we use RandomNumberGenerator service? your require a random number generator here

  edm::Service<RandomNumberGenerator> rng;
  if(!rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
            "which appears to be absent.  Please add that service to your configuration\n"
      "or remove the modules that require it." << std::endl;
  }
  // this is a global variable defined in GeneratorInterface/ExternalDecays/src/ExternalDecayDriver.cc
  ParticleReplacerGunVar::decayRandomEngine = &rng->getEngine();
  tauola_->SetDecayRandomEngine(ParticleReplacerGunVar::decayRandomEngine);

  if(forceTauPlusHelicity_ != 0) 
    edm::LogInfo("MuonReplacement") << "[ParticleReplacer::ParticleReplacer] "
                                    << "Forcing tau+ to helicity " << forceTauPlusHelicity_ << std::endl;
  if(forceTauMinusHelicity_ != 0)
    edm::LogInfo("MuonReplacement") << "[ParticleReplacer::ParticleReplacer] "
                                    << "Forcing tau- to helicity " << forceTauMinusHelicity_ << std::endl;
  if(forceTauPlusHelicity_ == 0 && forceTauMinusHelicity_ == 0)
    edm::LogInfo("MuonReplacement") << "[ParticleReplacer::ParticleReplacer] "
                                    << "     Forcing tau helicity as decayed from a " << forceTauPolarization_ << std::endl; 
  if(forceTauDecay_ != "" && forceTauDecay_ != "none")
    edm::LogInfo("MuonReplacement") << "[ParticleReplacer::ParticleReplacer] "
                                    << "Forcing tau decaying into " << forceTauDecay_ << std::endl;

  std::memset(pol1_, 0, 4*sizeof(float));
  std::memset(pol2_, 0, 4*sizeof(float));

  if(generatorMode_ != "Tauola")
    throw cms::Exception("Configuration") << "Generator mode other than Tauola is not supported" << std::endl;

  throw cms::Exception("UnimplementedFeature") << "ParticleReplacerParticleGun is not usable yet." << std::endl;
}

ParticleReplacerParticleGun::~ParticleReplacerParticleGun() {}

void ParticleReplacerParticleGun::beginJob() {
  gen::Pythia6Service::InstanceWrapper guard(&pythia_);

  pythia_.setGeneralParams();

  if(abs(gunParticle_) == 15) {
    /* FIXME
    call_tauola(-1,1);
    */
  }
}

void ParticleReplacerParticleGun::endJob() {
  if(abs(gunParticle_) == 15) {
    /* FIXME
       call_tauola(1,1);
    */
  }
}

std::auto_ptr<HepMC::GenEvent> ParticleReplacerParticleGun::produce(const reco::MuonCollection& muons, const reco::Vertex *pvtx, const HepMC::GenEvent *genEvt) {
  if(genEvt != 0)
    throw cms::Exception("UnimplementedFeature") << "ParticleReplacerParticleGun does NOT support merging at HepMC level" << std::endl;

  std::auto_ptr<HepMC::GenEvent> evt(0);
  std::vector<HepMC::FourVector> muons_corrected;
  muons_corrected.reserve(muons.size());
  correctTauMass(muons, muons_corrected);

  gen::Pythia6Service::InstanceWrapper guard(&pythia_);

  for(unsigned int i=0; i<muons_corrected.size(); ++i) {
    HepMC::FourVector& muon = muons_corrected[i];
    call_py1ent(i+1, gunParticle_*muons[i].charge(), muon.e(), muon.theta(), muon.phi());
  }

  // Let's not do a call_pyexec here because it's unnecessary

  if(printout_) {
    std::cout << " /////////////////////  After py1ent, before pyhepc /////////////////////" << std::endl;
    call_pylist(3);
  }

  // Vertex shift
  call_pyhepc(1); // pythia -> hepevt

  if(printout_) {
    std::cout << " /////////////////////  After pyhepc, before vertex shift /////////////////////" << std::endl;
    HepMC::HEPEVT_Wrapper::print_hepevt();
  }
  // Search for HepMC/HEPEVT_Wrapper.h for the wrapper interface
  int nparticles = HepMC::HEPEVT_Wrapper::number_entries();
  HepMC::ThreeVector shift(0,0,0); 
  if(particleOrigin_ == "primaryVertex") {
    if(!pvtx)
      throw cms::Exception("LogicError") << "Particle origin set to primaryVertex, but pvtx is null!" << std::endl;

    shift.setX(pvtx->x()*10); // cm -> mm
    shift.setY(pvtx->y()*10); // cm -> mm
    shift.setZ(pvtx->z()*10); // cm -> mm
  }
  for(int i=1; i <= nparticles; ++i) {
    if(abs(HepMC::HEPEVT_Wrapper::id(i)) != abs(gunParticle_)) {
      throw cms::Exception("LogicError") << "Particle in HEPEVT is " << HepMC::HEPEVT_Wrapper::id(i)
                                         << " is not the same as gunParticle " << gunParticle_
                                         << " for index " << i << std::endl;
    }

    if(particleOrigin_ == "muonReferencePoint") {
      const reco::Muon& muon = muons[i-1];
      shift.setX(muon.vx()*10); // cm -> mm
      shift.setY(muon.vy()*10); // cm -> mm
      shift.setZ(muon.vz()*10); // cm -> mm
    }

    HepMC::HEPEVT_Wrapper::set_position(i,
                                        HepMC::HEPEVT_Wrapper::x(i) + shift.x(),
                                        HepMC::HEPEVT_Wrapper::y(i) + shift.y(),
                                        HepMC::HEPEVT_Wrapper::z(i) + shift.z(),
                                        HepMC::HEPEVT_Wrapper::t(i));
  }

  if(printout_) {
    std::cout << " /////////////////////  After vertex shift, before pyhepc/tauola /////////////////////" << std::endl;
    HepMC::HEPEVT_Wrapper::print_hepevt();
  }

  if(abs(gunParticle_) == 15){
    // Code example from TauolaInterface::processEvent()
    /* FIXME
    int dummy = -1;
    int numGenParticles_beforeTAUOLA = call_ihepdim(dummy);
    */

    forceTauolaTauDecays();

    for(unsigned int i=0; i<muons_corrected.size(); ++i) {
      tauola_forParticleGun(i+1, gunParticle_*muons[i].charge(), muons_corrected[i]);
    }

    if(printout_) {
      std::cout << " /////////////////////  After tauola, before pyhepc  /////////////////////" << std::endl;
      HepMC::HEPEVT_Wrapper::print_hepevt();
    }
    call_pyhepc(2); // hepevt->pythia
    if(printout_) {
      std::cout << " /////////////////////  After pyhepc, before vertex fix  /////////////////////" << std::endl;
      call_pylist(3);
    }

    // Fix tau decay vertex position
    /* FIXME
    int numGenParticles_afterTAUOLA = call_ihepdim(dummy);
    tauola_.setDecayVertex(numGenParticles_beforeTAUOLA, numGenParticles_afterTAUOLA);
    */

    if(printout_) {
      /* FIXME
      std::cout << "     numGenParticles_beforeTAUOLA " << numGenParticles_beforeTAUOLA << std::endl
                << "     numGenParticles_afterTAUOLA  " << numGenParticles_afterTAUOLA << std::endl;
      */
      std::cout << " /////////////////////  After vertex fix, before pyexec  /////////////////////" << std::endl;
      call_pylist(3);
    }
  }
  else {
    call_pyhepc(2); // hepevt->pythia
    if(printout_) {
      std::cout << " /////////////////////  After pyhepc, before pyexec  /////////////////////" << std::endl;
      call_pylist(3);
    }
  }

  call_pyexec(); // taus: decay pi0's etc; others: decay whatever

  if(printout_) {
    std::cout << " /////////////////////  After pyexec, before pyhepc  /////////////////////" << std::endl;
    call_pylist(3);
  }

  call_pyhepc(1); // pythia -> hepevt

  HepMC::IO_HEPEVT conv;
  evt = std::auto_ptr<HepMC::GenEvent>(new HepMC::GenEvent(*conv.read_next_event()));

  if(printout_) {
    evt->print();

    std::cout << std::endl << "Vertices: " << std::endl;
    for(HepMC::GenEvent::vertex_const_iterator iter = evt->vertices_begin(); iter != evt->vertices_end(); ++iter) {
      std::cout << "Vertex " << (*iter)->id() << ", barcode " << (*iter)->barcode() << std::endl;

      HepMC::ThreeVector point = (*iter)->point3d();
      std::cout << "  Point (" << point.x() << ", " << point.y() << ", " << point.z() << ")" << std::endl;

      std::cout << "  Incoming particles: ";
      for(HepMC::GenVertex::particles_in_const_iterator pi = (*iter)->particles_in_const_begin(); pi != (*iter)->particles_in_const_end(); ++pi) {
        std::cout << (*pi)->pdg_id() << " ";
      }
      std::cout << std::endl;
      std::cout << "  Outgoing particles: ";
      for(HepMC::GenVertex::particles_out_const_iterator pi = (*iter)->particles_out_const_begin(); pi != (*iter)->particles_out_const_end(); ++pi) {
        std::cout << (*pi)->pdg_id() << " ";
      }
      std::cout << std::endl << std::endl;
    }
  }

  return evt;
}

void ParticleReplacerParticleGun::correctTauMass(const reco::MuonCollection& muons, std::vector<HepMC::FourVector>& corrected) {
  if(abs(gunParticle_) == 15) {
    // Correct energy for tau
    for(reco::MuonCollection::const_iterator iMuon = muons.begin(); iMuon != muons.end(); ++iMuon) {
      const reco::Muon::LorentzVector& vec = iMuon->p4();

      double E = sqrt(tauMass*tauMass + vec.x()*vec.x() + vec.y()*vec.y() + vec.z()*vec.z());

      corrected.push_back(HepMC::FourVector(vec.x(), vec.y(), vec.z(), E));
    }
  }
  else {
    // Just copy the LorentzVector
    for(reco::MuonCollection::const_iterator iMuon = muons.begin(); iMuon != muons.end(); ++iMuon) {
      corrected.push_back(iMuon->p4());
    }
  }
}

void ParticleReplacerParticleGun::forceTauolaTauDecays() {
  if(forceTauDecay_ == "" || forceTauDecay_ == "none") return;

  // for documentation look at Tauola file tauola_photos_ini.f
  // COMMON block COMMON / TAUBRA / GAMPRT(30),JLIST(30),NCHAN

  if(forceTauDecay_ == "hadrons"){
    /* FIXME
    taubra.gamprt[0] = 0; // disable branchings to electrons in Tauola
    taubra.gamprt[1] = 0; // disable branchings to muons in Tauola
    */
  }
  else if(forceTauDecay_ == "1prong"){
    /* FIXME
    taubra.gamprt[0]  = 0; // disable branchings to electrons in Tauola
    taubra.gamprt[1]  = 0; // disable branchings to muons in Tauola
    taubra.gamprt[7]  = 0;
    taubra.gamprt[9]  = 0;
    taubra.gamprt[10] = 0;
    taubra.gamprt[11] = 0;
    taubra.gamprt[12] = 0;
    taubra.gamprt[13] = 0;
    taubra.gamprt[17] = 0;
    */
  }
  else if(forceTauDecay_ == "3prong"){
    /* FIXME
    taubra.gamprt[0]  = 0; // disable branchings to electrons in Tauola
    taubra.gamprt[1]  = 0; // disable branchings to muons in Tauola
    taubra.gamprt[2]  = 0;
    taubra.gamprt[3]  = 0;
    taubra.gamprt[4]  = 0;
    taubra.gamprt[5]  = 0;
    taubra.gamprt[6]  = 0;
    taubra.gamprt[8]  = 0;
    taubra.gamprt[14] = 0;
    taubra.gamprt[15] = 0;
    taubra.gamprt[16] = 0;
    taubra.gamprt[18] = 0;
    taubra.gamprt[19] = 0;
    taubra.gamprt[20] = 0;
    taubra.gamprt[21] = 0;
    */
  }
  else 
    edm::LogError("MuonReplacement") << "[ParticleReplacerAlgoParticleGun::forceTauoladecays] "
                                     << "Unknown value for forcing tau decays: " << forceTauDecay_ << std::endl;
}

void ParticleReplacerParticleGun::tauola_forParticleGun(int tau_idx, int pdg_id, const HepMC::FourVector& particle_momentum) {
  if(abs(pdg_id) != 15) {
    edm::LogError("MuonReplacement") << "[ParticleReplacerAlgoParticleGun::tauola_forParticleGuns] "
                                     << "Trying to decay something else than tau: pdg_id = " << pdg_id << std::endl;
    return;
  }

  // By default the index of tau+ is 3 and tau- is 4. The TAUOLA
  // routine takes internally care of finding the correct
  // position of the tau which is decayed. However, since we are
  // calling DEXAY routine directly by ourselves, we must set
  // the index manually by ourselves. Fortunately, this seems to
  // be simple.
  /* FIXME
  if(printout_)
    std::cout << " Tauola taupos common block: np1 " << taupos.np1 << " np2 " << taupos.np2 << std::endl;
  taupos.np1 = tau_idx;
  taupos.np2 = tau_idx;
  if(printout_)
    std::cout << " Resetting taupos common block to: np1 " << taupos.np1 << " np2 " << taupos.np2 << std::endl;
  */

  if(pdg_id == -15){ // tau+

    pol1_[2] = tauHelicity(pdg_id);

    /* FIXME
    momdec.p1[0] = particle_momentum.x();
    momdec.p1[1] = particle_momentum.y();
    momdec.p1[2] = particle_momentum.z();
    momdec.p1[3] = particle_momentum.e();

    momdec.p2[0] = -momdec.p1[0];
    momdec.p2[1] = -momdec.p1[1];
    momdec.p2[2] = -momdec.p1[2];
    momdec.p2[3] =  momdec.p1[3];

    // "mother" p4
    momdec.q1[0] = 0;
    momdec.q1[1] = 0;
    momdec.q1[2] = 0;
    momdec.q1[3] = momdec.p1[3] + momdec.p2[3];

    call_dexay(1,pol1);
    */
  }
  else if (pdg_id == 15){ // tau-

    pol2_[2] = tauHelicity(pdg_id);

    /* FIXME
    momdec.p2[0] = particle_momentum.x();
    momdec.p2[1] = particle_momentum.y();
    momdec.p2[2] = particle_momentum.z();
    momdec.p2[3] = particle_momentum.e();

    momdec.p1[0] = -momdec.p2[0];
    momdec.p1[1] = -momdec.p2[1];
    momdec.p1[2] = -momdec.p2[2];
    momdec.p1[3] =  momdec.p2[3];

    // "mother" p4
    momdec.q1[0] = 0;
    momdec.q1[1] = 0;
    momdec.q1[2] = 0;
    momdec.q1[3] = momdec.p1[3] + momdec.p2[3];

    call_dexay(2,pol2);
    */
  }
}

float ParticleReplacerParticleGun::tauHelicity(int pdg_id) {
  /* tau polarization summary from Tauola source code:
     in all cases      W : pol1 = -1, pol2 = -1 (tau or neutrino)
     H+: pol1 = +1, pol2 = +1 (tau or neutrino)

     if neutral higgs    : pol1 = +1, pol2 = -1 OR pol1 = -1, pol2 = +1 (tau tau)
     if Z or undetermined: pol1 = +1, pol2 = +1 OR pol1 = -1, pol2 = -1 (tau tau)
  */

  if(pdg_id < 0) { // tau+
    if(forceTauPlusHelicity_ != 0) {
      return forceTauPlusHelicity_;
    }
    if(forceTauPolarization_ == "W") return -1;
    if(forceTauPolarization_ == "H+") return 1;
    if(pol2_[2] != 0) {
      if(forceTauPolarization_ == "h" ||
         forceTauPolarization_ == "H" ||
         forceTauPolarization_ == "A") return -pol2_[2];
      else return pol2_[2];
    }
    else {
      return randomPolarization(); //all other cases random, when first tau
    }
  }
  else {           // tau-
    if(forceTauMinusHelicity_ != 0) {
      return forceTauMinusHelicity_;
    }
    if(forceTauPolarization_ == "W") return -1;
    if(forceTauPolarization_ == "H+") return 1;
    if(pol1_[2] != 0){
      if(forceTauPolarization_ == "h" ||
         forceTauPolarization_ == "H" ||
         forceTauPolarization_ == "A") return -pol1_[2];
      else return pol1_[2];
    }
    else {
      return randomPolarization(); //all other cases random, when first tau
    }
  }

  edm::LogError("MuonReplacement") << "[ParticleReplacerAlgoParticleGun::tauHelicity] "
                                   << "tauHelicity undetermined, returning 0" << std::endl;
  return 0;
}

float ParticleReplacerParticleGun::randomPolarization() {
  uint32_t randomNumber = rand();
  if(randomNumber%2 > 0.5) return 1; 
  return -1;
}
