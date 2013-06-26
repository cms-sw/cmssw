#ifndef TtSemiLepSignalSelMVAComputer_h
#define TtSemiLepSignalSelMVAComputer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#ifndef TtSemiLepSignalSelMVARcd_defined  // to avoid conflicts with the TopSemiLepLepSignalSelMVATrainer
#define TtSemiLepSignalSelMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiLepSignalSelMVA);  // defines TopSemiLepLepSignalSelMVARcd
#endif

class TtSemiLepSignalSelMVAComputer : public edm::EDProducer {

 public:
  
  explicit TtSemiLepSignalSelMVAComputer(const edm::ParameterSet&);
  ~TtSemiLepSignalSelMVAComputer();
  
 private:

  virtual void beginJob();
  virtual void produce(edm::Event& evt, const edm::EventSetup& setup);
  virtual void endJob();

  double DeltaPhi(const math::XYZTLorentzVector& v1, const math::XYZTLorentzVector& v2);
  double DeltaR(const math::XYZTLorentzVector& v1, const math::XYZTLorentzVector& v2);

  edm::InputTag muons_;
  edm::InputTag jets_;
  edm::InputTag METs_;
  edm::InputTag electrons_;

  PhysicsTools::MVAComputerCache mvaComputer;

  double DiscSel;
  
};

#endif
