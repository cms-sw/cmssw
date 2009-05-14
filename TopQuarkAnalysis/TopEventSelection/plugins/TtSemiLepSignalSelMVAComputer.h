#ifndef TtSemiLepSignalSelMVAComputer_h
#define TtSemiLepSignalSelMVAComputer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#ifndef TtSemiLepSignalSelMVARcd_defined  // to avoid conflicts with the TopSemiLepLepSignalSelMVATrainer
#define TtSemiLepSignalSelMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtSemiLepSignalSelMVA);  // defines TopSemiLepLepSignalSelMVARcd
#endif

class TtSemiLepSignalSelMVAComputer : public edm::EDProducer {

 public:
  
  explicit TtSemiLepSignalSelMVAComputer(const edm::ParameterSet&);
  ~TtSemiLepSignalSelMVAComputer();
  
 private:

  virtual void beginJob(const edm::EventSetup&);
  virtual void produce(edm::Event& evt, const edm::EventSetup& setup);
  virtual void endJob();

  double DeltaPhi(math::XYZTLorentzVector v1, math::XYZTLorentzVector v2);
  double DeltaR(math::XYZTLorentzVector v1, math::XYZTLorentzVector v2);

  edm::InputTag muons_;
  edm::InputTag jets_;
  edm::InputTag METs_;
  edm::InputTag electrons_;

  PhysicsTools::MVAComputerCache mvaComputer;

  double DiscSel;
  
};

#endif
