/* L1CaloClusterFilter
Removes Cluster Overlap
and applies isolation criteria
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
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/FilteringModule.h"
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/IsolationModule.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

class L1CaloClusterIsolator : public edm::EDProducer {
   public:
      explicit L1CaloClusterIsolator(const edm::ParameterSet&);
      ~L1CaloClusterIsolator();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      /*INPUTS*/

      //Calorimeter Digis
      edm::InputTag clusters_;
      IsolationModule isolationModule;
};


using namespace edm;
using namespace std;
using namespace l1slhc;


L1CaloClusterIsolator::L1CaloClusterIsolator(const edm::ParameterSet& iConfig):
 clusters_(iConfig.getParameter<edm::InputTag>("src"))
{
  //Register Product
  produces<l1slhc::L1CaloClusterCollection>();
}


L1CaloClusterIsolator::~L1CaloClusterIsolator()
{

}



void
L1CaloClusterIsolator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::LogInfo ("INFO") << "Starting Cluster Filtering Algorithm" << endl;

  ESHandle<L1CaloTriggerSetup> setup;
  iSetup.get<L1CaloTriggerSetupRcd>().get(setup);

  isolationModule = IsolationModule(*setup);

   //Get ECAL + HCAL Digits from the EVent
   edm::Handle<L1CaloClusterCollection> clusters;
   iEvent.getByLabel(clusters_,clusters);

   //Book a temporary collection
   L1CaloClusterCollection tmp;

   //Book the Collection to be saved
   std::auto_ptr<L1CaloClusterCollection> isolatedClusters (new L1CaloClusterCollection);

   //Apply the algorithms
   isolationModule.isoDeposits(clusters,*isolatedClusters);

   //Put Clusters to file
   iEvent.put(isolatedClusters);

}

DEFINE_ANOTHER_FWK_MODULE(L1CaloClusterIsolator);
