#ifndef TtSemiLepSignalSelMVAComputer_h
#define TtSemiLepSignalSelMVAComputer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#ifndef TtSemiLepSignalSelMVARcd_defined  // to avoid conflicts with the TopSemiLepLepSignalSelMVATrainer
#define TtSemiLepSignalSelMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiLepSignalSelMVA);  // defines TopSemiLepLepSignalSelMVARcd
#endif

class TtSemiLepSignalSelMVAComputer : public edm::stream::EDProducer<> {
public:
  explicit TtSemiLepSignalSelMVAComputer(const edm::ParameterSet&);
  ~TtSemiLepSignalSelMVAComputer() override;

private:
  void produce(edm::Event& evt, const edm::EventSetup& setup) override;

  double DeltaPhi(const math::XYZTLorentzVector& v1, const math::XYZTLorentzVector& v2);
  double DeltaR(const math::XYZTLorentzVector& v1, const math::XYZTLorentzVector& v2);

  edm::ESGetToken<PhysicsTools::Calibration::MVAComputerContainer, TtSemiLepSignalSelMVARcd> mvaToken_;
  edm::EDGetTokenT<edm::View<pat::Muon> > muonsToken_;
  edm::EDGetTokenT<std::vector<pat::Jet> > jetsToken_;
  edm::EDGetTokenT<edm::View<pat::MET> > METsToken_;
  edm::EDGetTokenT<edm::View<pat::Electron> > electronsToken_;
  edm::EDPutTokenT<double> putToken_;

  PhysicsTools::MVAComputerCache mvaComputer;
};

#endif
