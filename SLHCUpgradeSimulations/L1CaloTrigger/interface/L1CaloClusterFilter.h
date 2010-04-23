/* L1CaloClusterFilter
Removes Cluster Overlap

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
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/CaloClusterFilteringCard.h"
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/CaloClusterIsolationCard.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

class L1CaloClusterFilter : public edm::EDProducer {
   public:
      explicit L1CaloClusterFilter(const edm::ParameterSet&);
      ~L1CaloClusterFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      /*INPUTS*/

      //Calorimeter Digis
      edm::InputTag clusters_;
      CaloClusterFilteringCard filterCard;
      CaloClusterIsolationCard isolCard;




};


