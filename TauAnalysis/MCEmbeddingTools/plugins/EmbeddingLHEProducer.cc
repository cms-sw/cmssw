// -*- C++ -*-
//
// Package:    test/EmbeddingLHEProducer
// Class:      EmbeddingLHEProducer
//
/**\class EmbeddingLHEProducer EmbeddingLHEProducer.cc test/EmbeddingLHEProducer/plugins/EmbeddingLHEProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Stefan Wayand
//         Created:  Wed, 13 Jan 2016 08:15:01 GMT
//
//

// system include files
#include <algorithm>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <memory>
#include "TLorentzVector.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEXMLStringProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "CLHEP/Random/RandExponential.h"

//
// class declaration
//

namespace CLHEP {
  class HepRandomEngine;
}

class EmbeddingLHEProducer : public edm::one::EDProducer<edm::BeginRunProducer, edm::EndRunProducer> {
public:
  explicit EmbeddingLHEProducer(const edm::ParameterSet &);
  ~EmbeddingLHEProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginJob() override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  void beginRunProduce(edm::Run &run, edm::EventSetup const &es) override;
  void endRunProduce(edm::Run &, edm::EventSetup const &) override;

  void fill_lhe_from_mumu(TLorentzVector &positiveLepton,
                          TLorentzVector &negativeLepton,
                          lhef::HEPEUP &outlhe,
                          CLHEP::HepRandomEngine *engine);
  void fill_lhe_with_particle(lhef::HEPEUP &outlhe, TLorentzVector &particle, int pdgid, double spin, double ctau);

  void transform_mumu_to_tautau(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton);
  const reco::Candidate *find_original_muon(const reco::Candidate *muon);
  void assign_4vector(TLorentzVector &Lepton, const pat::Muon *muon, std::string FSRmode);
  void mirror(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton);
  void InitialRecoCorrection(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton);
  void rotate180(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton);

  LHERunInfoProduct::Header give_slha();

  edm::EDGetTokenT<edm::View<pat::Muon>> muonsCollection_;
  edm::EDGetTokenT<reco::VertexCollection> vertexCollection_;
  int particleToEmbed_;
  bool mirror_, rotate180_,InitialRecoCorrection_;
  const double tauMass_ = 1.77682;
  const double muonMass_ = 0.1057;
  const double elMass_ = 0.00051;
  const int embeddingParticles[3]{11, 13, 15};

  std::ofstream file;
  bool write_lheout;

  // instead of reconstruted 4vectors of muons generated are taken to study FSR. Possible modes:
  // afterFSR - muons without FSR photons taken into account
  // beforeFSR - muons with FSR photons taken into account
  std::string studyFSRmode_;
};

//
// constructors and destructor
//
EmbeddingLHEProducer::EmbeddingLHEProducer(const edm::ParameterSet &iConfig) {
  //register your products
  produces<LHEEventProduct>();
  produces<LHERunInfoProduct, edm::Transition::BeginRun>();
  produces<math::XYZTLorentzVectorD>("vertexPosition");

  muonsCollection_ = consumes<edm::View<pat::Muon>>(iConfig.getParameter<edm::InputTag>("src"));
  vertexCollection_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"));
  particleToEmbed_ = iConfig.getParameter<int>("particleToEmbed");
  mirror_ = iConfig.getParameter<bool>("mirror");
  InitialRecoCorrection_ = iConfig.getParameter<bool>("InitialRecoCorrection");
  rotate180_ = iConfig.getParameter<bool>("rotate180");
  studyFSRmode_ = iConfig.getUntrackedParameter<std::string>("studyFSRmode", "");

  write_lheout = false;
  std::string lhe_ouputfile = iConfig.getUntrackedParameter<std::string>("lhe_outputfilename", "");
  if (!lhe_ouputfile.empty()) {
    write_lheout = true;
    file.open(lhe_ouputfile, std::fstream::out | std::fstream::trunc);
  }

  //check if particle can be embedded
  if (std::find(std::begin(embeddingParticles), std::end(embeddingParticles), particleToEmbed_) ==
      std::end(embeddingParticles)) {
    throw cms::Exception("Configuration") << "The given particle to embed is not in the list of allowed particles.";
  }

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration") << "The EmbeddingLHEProducer requires the RandomNumberGeneratorService\n"
                                             "which is not present in the configuration file. \n"
                                             "You must add the service\n"
                                             "in the configuration file or remove the modules that require it.";
  }
}

EmbeddingLHEProducer::~EmbeddingLHEProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void EmbeddingLHEProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine *engine = &rng->getEngine(iEvent.streamID());

  edm::Handle<edm::View<pat::Muon>> muonHandle;
  iEvent.getByToken(muonsCollection_, muonHandle);
  edm::View<pat::Muon> coll_muons = *muonHandle;

  Handle<std::vector<reco::Vertex>> coll_vertices;
  iEvent.getByToken(vertexCollection_, coll_vertices);

  TLorentzVector positiveLepton, negativeLepton;
  bool mu_plus_found = false;
  bool mu_minus_found = false;
  lhef::HEPEUP hepeup;
  hepeup.IDPRUP = 0;
  hepeup.XWGTUP = 1;
  hepeup.SCALUP = -1;
  hepeup.AQEDUP = -1;
  hepeup.AQCDUP = -1;
  // Assuming Pt-Order
  for (edm::View<pat::Muon>::const_iterator muon = coll_muons.begin(); muon != coll_muons.end(); ++muon) {
    if (muon->charge() == 1 && !mu_plus_found) {
      assign_4vector(positiveLepton, &(*muon), studyFSRmode_);
      mu_plus_found = true;
    } else if (muon->charge() == -1 && !mu_minus_found) {
      assign_4vector(negativeLepton, &(*muon), studyFSRmode_);
      mu_minus_found = true;
    } else if (mu_minus_found && mu_plus_found)
      break;
  }
  InitialRecoCorrection(positiveLepton,negativeLepton); //corrects Z mass peak to take into account smearing happening due to first muon reconstruction in the selection step
  mirror(positiveLepton, negativeLepton);                    // if no mirror, function does nothing.
  rotate180(positiveLepton, negativeLepton);                 // if no rotate180, function does nothing
  transform_mumu_to_tautau(positiveLepton, negativeLepton);  // if MuonEmbedding, function does nothing.
  fill_lhe_from_mumu(positiveLepton, negativeLepton, hepeup, engine);

  double originalXWGTUP_ = 1.;
  std::unique_ptr<LHEEventProduct> product(new LHEEventProduct(hepeup, originalXWGTUP_));

  if (write_lheout)
    std::copy(product->begin(), product->end(), std::ostream_iterator<std::string>(file));

  iEvent.put(std::move(product));
  // Saving vertex position
  std::unique_ptr<math::XYZTLorentzVectorD> vertex_position(
      new math::XYZTLorentzVectorD(coll_vertices->at(0).x(), coll_vertices->at(0).y(), coll_vertices->at(0).z(), 0.0));
  iEvent.put(std::move(vertex_position), "vertexPosition");
}

// ------------ method called once each job just before starting event loop  ------------
void EmbeddingLHEProducer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void EmbeddingLHEProducer::endJob() {}

// ------------ method called when starting to processes a run  ------------

void EmbeddingLHEProducer::beginRunProduce(edm::Run &run, edm::EventSetup const &) {
  // fill HEPRUP common block and store in edm::Run
  lhef::HEPRUP heprup;

  // set number of processes: 1 for Z to tau tau
  heprup.resize(1);

  //Process independent information

  //beam particles ID (two protons)
  //heprup.IDBMUP.first = 2212;
  //heprup.IDBMUP.second = 2212;

  //beam particles energies (both 6.5 GeV)
  //heprup.EBMUP.first = 6500.;
  //heprup.EBMUP.second = 6500.;

  //take default pdf group for both beamparticles
  //heprup.PDFGUP.first = -1;
  //heprup.PDFGUP.second = -1;

  //take certan pdf set ID (same as in officially produced DYJets LHE files)
  //heprup.PDFSUP.first = -1;
  //heprup.PDFSUP.second = -1;

  //master switch for event weight iterpretation (same as in officially produced DYJets LHE files)
  heprup.IDWTUP = 3;

  //Information for first process (Z to tau tau), for now only placeholder:
  heprup.XSECUP[0] = 1.;
  heprup.XERRUP[0] = 0;
  heprup.XMAXUP[0] = 1;
  heprup.LPRUP[0] = 1;

  std::unique_ptr<LHERunInfoProduct> runInfo(new LHERunInfoProduct(heprup));
  runInfo->addHeader(give_slha());

  if (write_lheout)
    std::copy(runInfo->begin(), runInfo->end(), std::ostream_iterator<std::string>(file));
  run.put(std::move(runInfo));
}

void EmbeddingLHEProducer::endRunProduce(edm::Run &run, edm::EventSetup const &es) {
  if (write_lheout) {
    file << LHERunInfoProduct::endOfFile();
    file.close();
  }
}

void EmbeddingLHEProducer::fill_lhe_from_mumu(TLorentzVector &positiveLepton,
                                              TLorentzVector &negativeLepton,
                                              lhef::HEPEUP &outlhe,
                                              CLHEP::HepRandomEngine *engine) {
  TLorentzVector Z = positiveLepton + negativeLepton;
  int leptonPDGID = particleToEmbed_;

  // double tau_ctau = 0.00871100; //cm
  double tau_ctau0 = 8.71100e-02;  // mm (for Pythia)
  double tau_ctau_p =
      tau_ctau0 * CLHEP::RandExponential::shoot(engine);  // return -std::log(HepRandom::getTheEngine()->flat());
  // replaces tau = process[iNow].tau0() * rndmPtr->exp(); from pythia8212/src/ProcessContainer.cc which is not initialized for ProcessLevel:all = off mode (no beam particle mode)
  double tau_ctau_n = tau_ctau0 * CLHEP::RandExponential::shoot(engine);
  //std::cout<<"tau_ctau P: "<<tau_ctau_p<<" tau_ctau N:  "<<tau_ctau_n<<std::endl;

  fill_lhe_with_particle(outlhe, Z, 23, 9.0, 0);
  fill_lhe_with_particle(outlhe, positiveLepton, -leptonPDGID, 1.0, tau_ctau_p);
  fill_lhe_with_particle(outlhe, negativeLepton, leptonPDGID, -1.0, tau_ctau_n);

  return;
}

void EmbeddingLHEProducer::fill_lhe_with_particle(
    lhef::HEPEUP &outlhe, TLorentzVector &particle, int pdgid, double spin, double ctau) {
  // Pay attention to different index conventions:
  // 'particleindex' follows usual C++ index conventions starting at 0 for a list.
  // 'motherindex' follows the LHE index conventions: 0 is for 'not defined', so the listing starts at 1.
  // That means: LHE index 1 == C++ index 0.
  int particleindex = outlhe.NUP;
  outlhe.resize(outlhe.NUP + 1);

  outlhe.PUP[particleindex][0] = particle.Px();
  outlhe.PUP[particleindex][1] = particle.Py();
  outlhe.PUP[particleindex][2] = particle.Pz();
  outlhe.PUP[particleindex][3] = particle.E();
  outlhe.PUP[particleindex][4] = particle.M();
  outlhe.IDUP[particleindex] = pdgid;
  outlhe.SPINUP[particleindex] = spin;
  outlhe.VTIMUP[particleindex] = ctau;

  outlhe.ICOLUP[particleindex].first = 0;
  outlhe.ICOLUP[particleindex].second = 0;

  if (std::abs(pdgid) == 23) {
    outlhe.MOTHUP[particleindex].first = 0;  // No Mother
    outlhe.MOTHUP[particleindex].second = 0;
    outlhe.ISTUP[particleindex] = 2;  // status
  }

  if (std::find(std::begin(embeddingParticles), std::end(embeddingParticles), std::abs(pdgid)) !=
      std::end(embeddingParticles)) {
    outlhe.MOTHUP[particleindex].first = 1;   // Mother is the Z (first partile)
    outlhe.MOTHUP[particleindex].second = 1;  // Mother is the Z (first partile)

    outlhe.ISTUP[particleindex] = 1;  //status
  }

  return;
}

void EmbeddingLHEProducer::transform_mumu_to_tautau(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton) {
  // No corrections applied for muon embedding
  double lep_mass;
  if (particleToEmbed_ == 11) {
    lep_mass = elMass_;
  } else if (particleToEmbed_ == 15) {
    lep_mass = tauMass_;
  } else {
    return;
  }

  TLorentzVector Z = positiveLepton + negativeLepton;

  TVector3 boost_from_Z_to_LAB = Z.BoostVector();
  TVector3 boost_from_LAB_to_Z = -Z.BoostVector();

  // Boosting the two leptons to Z restframe, then both are back to back. This means, same 3-momentum squared
  positiveLepton.Boost(boost_from_LAB_to_Z);
  negativeLepton.Boost(boost_from_LAB_to_Z);

  // Energy of tau = 0.5*Z-mass
  double lep_mass_squared = lep_mass * lep_mass;
  double lep_energy_squared = 0.25 * Z.M2();
  double lep_3momentum_squared = lep_energy_squared - lep_mass_squared;
  if (lep_3momentum_squared < 0) {
    edm::LogWarning("TauEmbedding") << "3-Momentum squared is negative";
    return;
  }

  //Computing scale, applying it on the 3-momenta and building new 4 momenta of the taus
  double scale = std::sqrt(lep_3momentum_squared / positiveLepton.Vect().Mag2());
  positiveLepton.SetPxPyPzE(scale * positiveLepton.Px(),
                            scale * positiveLepton.Py(),
                            scale * positiveLepton.Pz(),
                            std::sqrt(lep_energy_squared));
  negativeLepton.SetPxPyPzE(scale * negativeLepton.Px(),
                            scale * negativeLepton.Py(),
                            scale * negativeLepton.Pz(),
                            std::sqrt(lep_energy_squared));

  //Boosting the new taus back to LAB frame
  positiveLepton.Boost(boost_from_Z_to_LAB);
  negativeLepton.Boost(boost_from_Z_to_LAB);

  return;
}

void EmbeddingLHEProducer::assign_4vector(TLorentzVector &Lepton, const pat::Muon *muon, std::string FSRmode) {
  if ("afterFSR" == FSRmode && muon->genParticle() != nullptr) {
    const reco::GenParticle *afterFSRMuon = muon->genParticle();
    Lepton.SetPxPyPzE(
        afterFSRMuon->p4().px(), afterFSRMuon->p4().py(), afterFSRMuon->p4().pz(), afterFSRMuon->p4().e());
  } else if ("beforeFSR" == FSRmode && muon->genParticle() != nullptr) {
    const reco::Candidate *beforeFSRMuon = find_original_muon(muon->genParticle());
    Lepton.SetPxPyPzE(
        beforeFSRMuon->p4().px(), beforeFSRMuon->p4().py(), beforeFSRMuon->p4().pz(), beforeFSRMuon->p4().e());
  } else {
    Lepton.SetPxPyPzE(muon->p4().px(), muon->p4().py(), muon->p4().pz(), muon->p4().e());
  }
  return;
}

const reco::Candidate *EmbeddingLHEProducer::find_original_muon(const reco::Candidate *muon) {
  if (muon->mother(0) == nullptr)
    return muon;
  if (muon->pdgId() == muon->mother(0)->pdgId())
    return find_original_muon(muon->mother(0));
  else
    return muon;
}

void EmbeddingLHEProducer::rotate180(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton) {
  if (!rotate180_)
    return;
  edm::LogInfo("TauEmbedding") << "Applying 180<C2><B0> rotation";
  // By construction, the 3-momenta of mu-, mu+ and Z are in one plane.
  // That means, one vector for perpendicular projection can be used for both leptons.
  TLorentzVector Z = positiveLepton + negativeLepton;

  edm::LogInfo("TauEmbedding") << "MuMinus before. Pt: " << negativeLepton.Pt() << " Eta: " << negativeLepton.Eta()
                               << " Phi: " << negativeLepton.Phi() << " Mass: " << negativeLepton.M();

  TVector3 Z3 = Z.Vect();
  TVector3 positiveLepton3 = positiveLepton.Vect();
  TVector3 negativeLepton3 = negativeLepton.Vect();

  TVector3 p3_perp = positiveLepton3 - positiveLepton3.Dot(Z3) / Z3.Dot(Z3) * Z3;
  p3_perp = p3_perp.Unit();

  positiveLepton3 -= 2 * positiveLepton3.Dot(p3_perp) * p3_perp;
  negativeLepton3 -= 2 * negativeLepton3.Dot(p3_perp) * p3_perp;

  positiveLepton.SetVect(positiveLepton3);
  negativeLepton.SetVect(negativeLepton3);

  edm::LogInfo("TauEmbedding") << "MuMinus after. Pt: " << negativeLepton.Pt() << " Eta: " << negativeLepton.Eta()
                               << " Phi: " << negativeLepton.Phi() << " Mass: " << negativeLepton.M();

  return;
}

void EmbeddingLHEProducer::InitialRecoCorrection(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton)
{
  if(!InitialRecoCorrection_) return;
  edm::LogInfo("TauEmbedding")<< "Applying initial reconstruction correction" ;
  TLorentzVector Z = positiveLepton + negativeLepton;

  edm::LogInfo("TauEmbedding") << "MuMinus before. Pt: " << negativeLepton.Pt() << " Mass: " << negativeLepton.M() ;
  //std::cout << " MuMinus before. Pt: " << negativeLepton.Pt() << " Mass: " << negativeLepton.M() << " Energy: " << negativeLepton.E() << std::endl;
  float diLeptonMass=(positiveLepton + negativeLepton).M();
  if(diLeptonMass>60.&&diLeptonMass<122.) {
    //std::cout << "DiLeptonMass " << diLeptonMass << std::endl;
    float zmass=91.1876;
    float correction_deviation=5.; // to ensure only a correction that drops corresponding to a Gaussian with mean zmass and std. dev. 5 GeV
    double EmbeddingCorrection = 1.138; // value derived by function fitting to fold embedded mass spectrum back to original selection when using mu -> mu embedding
    EmbeddingCorrection=EmbeddingCorrection/(EmbeddingCorrection-(EmbeddingCorrection-1.)*exp(-pow((diLeptonMass-zmass),2.)/(2.*pow(correction_deviation,2.))));
    EmbeddingCorrection=((diLeptonMass + (EmbeddingCorrection - 1.)*zmass)/(diLeptonMass*EmbeddingCorrection));
    double correctedpositiveLeptonEnergy=std::sqrt(muonMass_*muonMass_+EmbeddingCorrection*positiveLepton.Px()*EmbeddingCorrection*positiveLepton.Px()+EmbeddingCorrection*positiveLepton.Py()*EmbeddingCorrection*positiveLepton.Py()+EmbeddingCorrection*positiveLepton.Pz()*EmbeddingCorrection*positiveLepton.Pz());
    double correctednegativeLeptonEnergy=std::sqrt(muonMass_*muonMass_+EmbeddingCorrection*negativeLepton.Px()*EmbeddingCorrection*negativeLepton.Px()+EmbeddingCorrection*negativeLepton.Py()*EmbeddingCorrection*negativeLepton.Py()+EmbeddingCorrection*negativeLepton.Pz()*EmbeddingCorrection*negativeLepton.Pz());
    positiveLepton.SetPxPyPzE(EmbeddingCorrection*positiveLepton.Px(),EmbeddingCorrection*positiveLepton.Py(),EmbeddingCorrection*positiveLepton.Pz(),correctedpositiveLeptonEnergy);
    negativeLepton.SetPxPyPzE(EmbeddingCorrection*negativeLepton.Px(),EmbeddingCorrection*negativeLepton.Py(),EmbeddingCorrection*negativeLepton.Pz(),correctednegativeLeptonEnergy);

    edm::LogInfo("TauEmbedding") << "MuMinus after. Pt: " << negativeLepton.Pt() << " Mass: " << negativeLepton.M() ;
    //std::cout << " MuMinus after. Pt: " << negativeLepton.Pt() << " Mass: " << negativeLepton.M() << " Energy: " << negativeLepton.E() << std::endl;
  }
  return;
}

void EmbeddingLHEProducer::mirror(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton) {
  if (!mirror_)
    return;
  edm::LogInfo("TauEmbedding") << "Applying mirroring";
  TLorentzVector Z = positiveLepton + negativeLepton;

  edm::LogInfo("TauEmbedding") << "MuMinus before. Pt: " << negativeLepton.Pt() << " Eta: " << negativeLepton.Eta()
                               << " Phi: " << negativeLepton.Phi() << " Mass: " << negativeLepton.M();

  TVector3 Z3 = Z.Vect();
  TVector3 positiveLepton3 = positiveLepton.Vect();
  TVector3 negativeLepton3 = negativeLepton.Vect();

  TVector3 beam(0., 0., 1.);
  TVector3 perpToZandBeam = Z3.Cross(beam).Unit();

  positiveLepton3 -= 2 * positiveLepton3.Dot(perpToZandBeam) * perpToZandBeam;
  negativeLepton3 -= 2 * negativeLepton3.Dot(perpToZandBeam) * perpToZandBeam;

  positiveLepton.SetVect(positiveLepton3);
  negativeLepton.SetVect(negativeLepton3);

  edm::LogInfo("TauEmbedding") << "MuMinus after. Pt: " << negativeLepton.Pt() << " Eta: " << negativeLepton.Eta()
                               << " Phi: " << negativeLepton.Phi() << " Mass: " << negativeLepton.M();

  return;
}

LHERunInfoProduct::Header EmbeddingLHEProducer::give_slha() {
  LHERunInfoProduct::Header slhah("slha");

  slhah.addLine("######################################################################\n");
  slhah.addLine("## PARAM_CARD AUTOMATICALY GENERATED BY MG5 FOLLOWING UFO MODEL   ####\n");
  slhah.addLine("######################################################################\n");
  slhah.addLine("##                                                                  ##\n");
  slhah.addLine("##  Width set on Auto will be computed following the information    ##\n");
  slhah.addLine("##        present in the decay.py files of the model.               ##\n");
  slhah.addLine("##        See  arXiv:1402.1178 for more details.                    ##\n");
  slhah.addLine("##                                                                  ##\n");
  slhah.addLine("######################################################################\n");
  slhah.addLine("\n");
  slhah.addLine("###################################\n");
  slhah.addLine("## INFORMATION FOR MASS\n");
  slhah.addLine("###################################\n");
  slhah.addLine("Block mass \n");
  slhah.addLine("    6 1.730000e+02 # MT \n");
  slhah.addLine("   15 1.777000e+00 # MTA \n");
  slhah.addLine("   23 9.118800e+01 # MZ \n");
  slhah.addLine("   25 1.250000e+02 # MH \n");
  slhah.addLine("## Dependent parameters, given by model restrictions.\n");
  slhah.addLine("## Those values should be edited following the \n");
  slhah.addLine("## analytical expression. MG5 ignores those values \n");
  slhah.addLine("## but they are important for interfacing the output of MG5\n");
  slhah.addLine("## to external program such as Pythia.\n");
  slhah.addLine("  1 0.000000 # d : 0.0 \n");
  slhah.addLine("  2 0.000000 # u : 0.0 \n");
  slhah.addLine("  3 0.000000 # s : 0.0 \n");
  slhah.addLine("  4 0.000000 # c : 0.0 \n");
  slhah.addLine("  5 0.000000 # b : 0.0 \n");
  slhah.addLine("  11 0.000000 # e- : 0.0 \n");
  slhah.addLine("  12 0.000000 # ve : 0.0 \n");
  slhah.addLine("  13 0.000000 # mu- : 0.0 \n");
  slhah.addLine("  14 0.000000 # vm : 0.0 \n");
  slhah.addLine("  16 0.000000 # vt : 0.0 \n");
  slhah.addLine("  21 0.000000 # g : 0.0 \n");
  slhah.addLine("  22 0.000000 # a : 0.0 \n");
  slhah.addLine(
      "  24 80.419002 # w+ : cmath.sqrt(MZ__exp__2/2. + cmath.sqrt(MZ__exp__4/4. - "
      "(aEW*cmath.pi*MZ__exp__2)/(Gf*sqrt__2))) \n");
  slhah.addLine("\n");
  slhah.addLine("###################################\n");
  slhah.addLine("## INFORMATION FOR SMINPUTS\n");
  slhah.addLine("###################################\n");
  slhah.addLine("Block sminputs \n");
  slhah.addLine("    1 1.325070e+02 # aEWM1 \n");
  slhah.addLine("    2 1.166390e-05 # Gf \n");
  slhah.addLine("    3 1.180000e-01 # aS \n");
  slhah.addLine("\n");
  slhah.addLine("###################################\n");
  slhah.addLine("## INFORMATION FOR WOLFENSTEIN\n");
  slhah.addLine("###################################\n");
  slhah.addLine("Block wolfenstein \n");
  slhah.addLine("    1 2.253000e-01 # lamWS \n");
  slhah.addLine("    2 8.080000e-01 # AWS \n");
  slhah.addLine("    3 1.320000e-01 # rhoWS \n");
  slhah.addLine("    4 3.410000e-01 # etaWS \n");
  slhah.addLine("\n");
  slhah.addLine("###################################\n");
  slhah.addLine("## INFORMATION FOR YUKAWA\n");
  slhah.addLine("###################################\n");
  slhah.addLine("Block yukawa \n");
  slhah.addLine("    6 1.730000e+02 # ymt \n");
  slhah.addLine("   15 1.777000e+00 # ymtau \n");
  slhah.addLine("\n");
  slhah.addLine("###################################\n");
  slhah.addLine("## INFORMATION FOR DECAY\n");
  slhah.addLine("###################################\n");
  slhah.addLine("DECAY   6 1.491500e+00 # WT \n");
  slhah.addLine("DECAY  15 2.270000e-12 # WTau \n");
  slhah.addLine("DECAY  23 2.441404e+00 # WZ \n");
  slhah.addLine("DECAY  24 2.047600e+00 # WW \n");
  slhah.addLine("DECAY  25 6.382339e-03 # WH \n");
  slhah.addLine("## Dependent parameters, given by model restrictions.\n");
  slhah.addLine("## Those values should be edited following the \n");
  slhah.addLine("## analytical expression. MG5 ignores those values \n");
  slhah.addLine("## but they are important for interfacing the output of MG5\n");
  slhah.addLine("## to external program such as Pythia.\n");
  slhah.addLine("DECAY  1 0.000000 # d : 0.0 \n");
  slhah.addLine("DECAY  2 0.000000 # u : 0.0 \n");
  slhah.addLine("DECAY  3 0.000000 # s : 0.0 \n");
  slhah.addLine("DECAY  4 0.000000 # c : 0.0 \n");
  slhah.addLine("DECAY  5 0.000000 # b : 0.0 \n");
  slhah.addLine("DECAY  11 0.000000 # e- : 0.0 \n");
  slhah.addLine("DECAY  12 0.000000 # ve : 0.0 \n");
  slhah.addLine("DECAY  13 0.000000 # mu- : 0.0 \n");
  slhah.addLine("DECAY  14 0.000000 # vm : 0.0 \n");
  slhah.addLine("DECAY  16 0.000000 # vt : 0.0 \n");
  slhah.addLine("DECAY  21 0.000000 # g : 0.0 \n");
  slhah.addLine("DECAY  22 0.000000 # a : 0.0\n");

  return slhah;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void EmbeddingLHEProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EmbeddingLHEProducer);
