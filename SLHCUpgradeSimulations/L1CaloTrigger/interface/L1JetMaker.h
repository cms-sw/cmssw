/* L1JetMaker
Produces the Jet Collection
M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/CaloJetCard.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

class L1JetMaker : public edm::EDProducer {
   public:
      explicit L1JetMaker(const edm::ParameterSet&);
      ~L1JetMaker();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      /*INPUTS*/

      //Filtered Clusters
      edm::InputTag clusters_;

      CaloJetCard card;



};


