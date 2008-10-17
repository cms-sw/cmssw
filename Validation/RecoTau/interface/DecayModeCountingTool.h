#ifndef DecayModeCountingTool_h
#define DecayModeCountingTool_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>
#include <string>
#include "TLorentzVector.h"

#include "TH1D.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


typedef math::XYZTLorentzVectorD LorentzVector;
typedef std::vector<LorentzVector> LorentzVectorCollection;

class DecayModeCountingTool : public edm::EDAnalyzer {
  
public:
  explicit DecayModeCountingTool(const edm::ParameterSet&);
  ~DecayModeCountingTool() {;}

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup& );
  virtual void endJob();

 private:

  std::vector<reco::GenParticle*> getGenStableDecayProducts(const reco::GenParticle * particle);

  enum tauDecayModes {kElectron, kMuon, 
		      kOneProng0pi0, kOneProng1pi0, kOneProng2pi0,
		      kThreeProng0pi0, kThreeProng1pi0,
		      kOther, kUndefined};

  MonitorElement* hGenTauDecay_DecayModes_;


  edm::InputTag MC_;
  double ptMinMCTau_;
  double ptMinMCElectron_;
  double ptMinMCMuon_;
  std::vector<int> m_PDG_;
  double etaMax;

 // output histograms
  bool saveoutputhistograms_;

  std::string tversion;
  std::string outPutFile_;

  DQMStore* dbeDecay;

};

#endif
