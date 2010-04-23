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
#include "L1Trigger/SLHCCaloTrigger/interface/JetCard.h"
#include "CondFormats/L1SLHCCaloTrigger/interface/L1CaloTriggerSetup.h"
#include "CondFormats/L1SLHCCaloTrigger/interface/L1CaloTriggerSetupRcd.h"
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
      
      JetCard card;



};


