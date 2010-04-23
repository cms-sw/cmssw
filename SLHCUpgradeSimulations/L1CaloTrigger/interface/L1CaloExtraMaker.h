/* L1ExtraMaker
Creates L1 Extra Objects from Clusters and jets

M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1SLHCCaloTrigger/interface/L1CaloCluster.h"
#include "DataFormats/L1SLHCCaloTrigger/interface/L1CaloJet.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

class L1CaloGeometry;

class L1ExtraMaker : public edm::EDProducer {
   public:
      explicit L1ExtraMaker(const edm::ParameterSet&);
      ~L1ExtraMaker();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;



      edm::InputTag clusters_;
      edm::InputTag jets_;

      int nObjects_; //Number of Objects to produce


};


