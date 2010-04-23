#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloClusterFilter.h"

   using namespace edm;
   using namespace std;

   using namespace l1slhc;


L1CaloClusterFilter::L1CaloClusterFilter(const edm::ParameterSet& iConfig):
 clusters_(iConfig.getParameter<edm::InputTag>("CrudeClusters"))
{
  //Register Product
  produces<l1slhc::L1CaloClusterCollection>("FilteredClusters");
  produces<l1slhc::L1CaloClusterCollection>("ParticleClusters");
}


L1CaloClusterFilter::~L1CaloClusterFilter()
{

}



void
L1CaloClusterFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::LogInfo ("INFO") << "Starting Cluster Filtering Algorithm" << endl;

  ESHandle<L1CaloTriggerSetup> setup;
  iSetup.get<L1CaloTriggerSetupRcd>().get(setup);

  filterCard = CaloClusterFilteringCard(*setup);
  isolCard = CaloClusterIsolationCard(*setup);

   //Get ECAL + HCAL Digits from the EVent
   edm::Handle<L1CaloClusterCollection> clusters;
   iEvent.getByLabel(clusters_,clusters);

   //Book a temporary collection
   L1CaloClusterCollection tmp;

   //Book the Collection to be saved
   std::auto_ptr<L1CaloClusterCollection> filteredClusters (new L1CaloClusterCollection);
   std::auto_ptr<L1CaloClusterCollection> isolatedClusters (new L1CaloClusterCollection);

   //Apply the algorithms
   filterCard.cleanClusters(*clusters,*filteredClusters);
   isolCard.isoDeposits(*filteredClusters,*isolatedClusters);


   //Put Clusters to file
   std::sort(filteredClusters->begin(),filteredClusters->end(),HigherClusterEt());
   std::sort(isolatedClusters->begin(),isolatedClusters->end(),HigherClusterEt());

   iEvent.put(filteredClusters,"FilteredClusters");
   iEvent.put(isolatedClusters,"ParticleClusters");

}

// ------------ method called once each job just before starting event loop  ------------
void
L1CaloClusterFilter::beginJob(const edm::EventSetup& setup)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1CaloClusterFilter::endJob() {
}



