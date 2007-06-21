#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

using namespace std;

class StGenEventReco : public edm::EDProducer {
   public:
      explicit StGenEventReco(const edm::ParameterSet&);
      ~StGenEventReco();

      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
};
