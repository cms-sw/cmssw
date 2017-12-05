#ifndef TtSemiLepSignalSelMVATrainer_h
#define TtSemiLepSignalSelMVATrainer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#ifndef TtSemiLepSignalSelMVARcd_defined  // to avoid conflicts with the TtSemiSignalSelMVAComputer
#define TtSemiLepSignalSelMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiLepSignalSelMVA);  // defines TtSemiLepSignalSelMVA
#endif

class TtSemiLepSignalSelMVATrainer : public edm::EDAnalyzer {

 public:

  explicit TtSemiLepSignalSelMVATrainer(const edm::ParameterSet&);
  ~TtSemiLepSignalSelMVATrainer() override;

 private:

  void analyze(const edm::Event& evt, const edm::EventSetup& setup) override;
  void beginJob() override;

  double DeltaPhi(const math::XYZTLorentzVector& v1,const math::XYZTLorentzVector& v2);
  double DeltaR(const math::XYZTLorentzVector& v1,const math::XYZTLorentzVector& v2);

  // pt sorting stuff
  struct JetwithHigherPt {
    bool operator() ( const pat::Jet& j1, const pat::Jet& j2) const {
      return j1.pt() > j2.pt();
    };
  };

  edm::EDGetTokenT< edm::View<pat::Muon> > muonsToken_;
  edm::EDGetTokenT< edm::View<pat::Electron> > electronsToken_;
  edm::EDGetTokenT< std::vector<pat::Jet> > jetsToken_;
  edm::EDGetTokenT<edm::View<pat::MET> > METsToken_;
  edm::EDGetTokenT<TtGenEvent> genEvtToken_;

  int lepChannel_;
  int whatData_;
  int maxEv_;
  int selEv;

  PhysicsTools::MVAComputerCache mvaComputer;

};

#endif
