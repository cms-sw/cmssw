#ifndef RecoMuonFromPFProducer_h_
#define RecoMuonFromPFProducer_h_


// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

// useful?
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**\class  RecoMuonFromPFProducer
\brief Produces a collection of reco::Muons referred to by the 
PFCandidates of type muons in a collection of PFCandidates. 
*/

class RecoMuonFromPFProducer : public edm::EDProducer {
 public:
  explicit RecoMuonFromPFProducer(const edm::ParameterSet&);
  ~RecoMuonFromPFProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void beginJob();
  virtual void beginRun(edm::Run &, const edm::EventSetup &);

 private:

  edm::InputTag  inputTagPF_;

  /// verbose ?
  bool  verbose_;
};

#endif
