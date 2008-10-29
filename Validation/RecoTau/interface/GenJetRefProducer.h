/*GenJetRefProducer
Producer that creates LorentzVector Collections
from generator level jets
*/

#ifndef GenJetRefProducer_h
#define GenJetRefProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include <vector>
#include <string>

typedef math::XYZTLorentzVectorD LorentzVector;
typedef std::vector<LorentzVector> LorentzVectorCollection;

class GenJetRefProducer : public edm::EDProducer {
  
public:
  explicit GenJetRefProducer(const edm::ParameterSet&);
  ~GenJetRefProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:


  edm::InputTag genJetSrc_;
  double ptMinGenJet_;
  double etaMax;

};

#endif
