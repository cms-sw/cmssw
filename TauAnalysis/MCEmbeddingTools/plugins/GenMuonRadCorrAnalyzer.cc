#include "TauAnalysis/MCEmbeddingTools/plugins/GenMuonRadCorrAnalyzer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <TMath.h>

GenMuonRadCorrAnalyzer::GenMuonRadCorrAnalyzer(const edm::ParameterSet& cfg)
  : beamEnergy_(cfg.getParameter<double>("beamEnergy")),
    muonRadiationAlgo_(0),
    numWarnings_(0)
{
  srcSelectedMuons_ = cfg.getParameter<edm::InputTag>("srcSelectedMuons"); 
  srcGenParticles_  = cfg.getParameter<edm::InputTag>("srcGenParticles"); 

  srcWeights_ = cfg.getParameter<vInputTag>("srcWeights");

  directory_ = cfg.getParameter<std::string>("directory");

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
  
  typedef std::vector<double> vdouble;
  vdouble binningMuonEn = cfg.getParameter<vdouble>("binningMuonEn");
  int numBinsMuonEn = binningMuonEn.size() - 1;
  if ( !(numBinsMuonEn >= 1) ) throw cms::Exception("Configuration")
    << " Invalid Configuration Parameter 'binningMuonEn', must define at least one bin !!\n";
  
  unsigned numBinsRadDivMuonEn = cfg.getParameter<unsigned>("numBinsRadDivMuonEn");
  double minRadDivMuonEn = cfg.getParameter<double>("minRadDivMuonEn");
  double maxRadDivMuonEn = cfg.getParameter<double>("maxRadDivMuonEn");

  for ( int iBinMuPlusEn = 0; iBinMuPlusEn < numBinsMuonEn; ++iBinMuPlusEn ) {
    double minMuPlusEn = binningMuonEn[iBinMuPlusEn];
    double maxMuPlusEn = binningMuonEn[iBinMuPlusEn + 1];
    for ( int iBinMuMinusEn = 0; iBinMuMinusEn < numBinsMuonEn; ++iBinMuMinusEn ) {
      double minMuMinusEn = binningMuonEn[iBinMuMinusEn];
      double maxMuMinusEn = binningMuonEn[iBinMuMinusEn + 1];
      plotEntryType* plotEntry = 
        new plotEntryType(minMuPlusEn, maxMuPlusEn, minMuMinusEn, maxMuMinusEn, 
			  numBinsRadDivMuonEn, minRadDivMuonEn, maxRadDivMuonEn);
      plotEntries_.push_back(plotEntry);
    }
  }

  std::string muonRadiationAlgo_string = cfg.getParameter<std::string>("muonRadiationAlgo");
  if ( muonRadiationAlgo_string == "" ) {
    muonRadiationAlgo_ = 0;
  } else if ( muonRadiationAlgo_string == "pythia" ) {
    edm::ParameterSet cfgMuonRadiationAlgo_pythia(cfg);
    cfgMuonRadiationAlgo_pythia.addParameter<std::string>("mode", "pythia");
    cfgMuonRadiationAlgo_pythia.addParameter<int>("verbosity", verbosity_);
    muonRadiationAlgo_ = new GenMuonRadiationAlgorithm(cfgMuonRadiationAlgo_pythia);
  } else if ( muonRadiationAlgo_string == "photos" ) {
    edm::ParameterSet cfgMuonRadiationAlgo_photos(cfg);
    cfgMuonRadiationAlgo_photos.addParameter<std::string>("mode", "photos");
    cfgMuonRadiationAlgo_photos.addParameter<int>("verbosity", verbosity_);
    muonRadiationAlgo_ = new GenMuonRadiationAlgorithm(cfgMuonRadiationAlgo_photos);
  } else throw cms::Exception("Configuration")
      << " Invalid Configuration Parameter 'muonRadiationAlgo' = " << muonRadiationAlgo_string << " !!\n";
}

GenMuonRadCorrAnalyzer::~GenMuonRadCorrAnalyzer()
{
  for ( std::vector<plotEntryType*>::iterator it = plotEntries_.begin();
	it != plotEntries_.end(); ++it ) {
    delete (*it);
  }

  delete muonRadiationAlgo_;
}
    
void GenMuonRadCorrAnalyzer::beginJob()
{
  edm::Service<TFileService> fs;

  TFileDirectory dir = ( directory_ != "" ) ? fs->mkdir(directory_) : (fs->tFileDirectory());
  for ( std::vector<plotEntryType*>::iterator plotEntry = plotEntries_.begin();
	plotEntry != plotEntries_.end(); ++plotEntry ) {
    (*plotEntry)->bookHistograms(dir);
  }
}

void GenMuonRadCorrAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& es)
{
  double evtWeight = 1.0;
  for ( vInputTag::const_iterator srcWeight = srcWeights_.begin();
	srcWeight != srcWeights_.end(); ++srcWeight ) {
    edm::Handle<double> weight;
    evt.getByLabel(*srcWeight, weight);
    evtWeight *= (*weight);
  }

  if ( evtWeight < 1.e-3 || evtWeight > 1.e+3 || TMath::IsNaN(evtWeight) ) return;

  std::vector<reco::CandidateBaseRef> selectedMuons = getSelMuons(evt, srcSelectedMuons_);
  
  edm::Handle<reco::GenParticleCollection> genParticles;
  evt.getByLabel(srcGenParticles_, genParticles);

  reco::Candidate::LorentzVector genMuonPlusP4_beforeRad;
  reco::Candidate::LorentzVector genMuonPlusP4_afterRad;
  bool genMuonPlus_found = false;
  reco::Candidate::LorentzVector genMuonMinusP4_beforeRad;
  reco::Candidate::LorentzVector genMuonMinusP4_afterRad;
  bool genMuonMinus_found = false;

  std::vector<int> muonPdgIds;
  muonPdgIds.push_back(-13);
  muonPdgIds.push_back(+13);

  for ( std::vector<reco::CandidateBaseRef>::const_iterator selectedMuon = selectedMuons.begin();
	selectedMuon != selectedMuons.end(); ++selectedMuon ) {
    const reco::GenParticle* genMuon_matched = findGenParticleForMCEmbedding((*selectedMuon)->p4(), *genParticles, 0.3, -1, &muonPdgIds, true);
    if ( genMuon_matched && genMuon_matched->charge() > +0.5 ) {
      genMuonPlusP4_beforeRad = genMuon_matched->p4();
      genMuonPlusP4_afterRad = genMuonPlusP4_beforeRad;
      compGenMuonP4afterRad(genMuon_matched, genMuonPlusP4_afterRad);
      genMuonPlus_found = true;
    }
    if ( genMuon_matched && genMuon_matched->charge() < -0.5 ) {
      genMuonMinusP4_beforeRad = genMuon_matched->p4();
      genMuonMinusP4_afterRad = genMuonMinusP4_beforeRad;
      compGenMuonP4afterRad(genMuon_matched, genMuonMinusP4_afterRad);
      genMuonMinus_found = true;
    }
  }

  if ( !(genMuonPlus_found && genMuonMinus_found) ) return;

  double muonPlusRad = 0.;
  int muonPlusRad_error = 0;
  double muonMinusRad = 0.;
  int muonMinusRad_error = 0;
  if ( muonRadiationAlgo_ ) {
    muonPlusRad = muonRadiationAlgo_->compFSR(evt.streamID(), genMuonPlusP4_beforeRad, +1, genMuonMinusP4_beforeRad, muonPlusRad_error).E();
    muonMinusRad = muonRadiationAlgo_->compFSR(evt.streamID(), genMuonMinusP4_beforeRad, -1, genMuonPlusP4_beforeRad, muonMinusRad_error).E();
  } else {
    muonPlusRad = genMuonPlusP4_beforeRad.E() - genMuonPlusP4_afterRad.E();
    muonMinusRad = genMuonMinusP4_beforeRad.E() - genMuonMinusP4_afterRad.E();
  }

  if ( muonPlusRad_error || muonMinusRad_error ) return;

  for ( std::vector<plotEntryType*>::iterator plotEntry = plotEntries_.begin();
	plotEntry != plotEntries_.end(); ++plotEntry ) {
    (*plotEntry)->fillHistograms(genMuonPlusP4_afterRad.E(), muonPlusRad, genMuonMinusP4_afterRad.E(), muonMinusRad, evtWeight);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenMuonRadCorrAnalyzer);
