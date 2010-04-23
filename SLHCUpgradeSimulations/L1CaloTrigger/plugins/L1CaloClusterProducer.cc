#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloClusterProducer.h"

   using namespace edm;
   using namespace std;

   using namespace l1slhc;


L1CaloClusterProducer::L1CaloClusterProducer(const edm::ParameterSet& iConfig):
  towers_(iConfig.getParameter<edm::InputTag>("L1Towers"))
{
  //Register Product
  produces<l1slhc::L1CaloClusterCollection>();
}


L1CaloClusterProducer::~L1CaloClusterProducer()
{

}



void
L1CaloClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::LogInfo("ClusterProducer") << "Starting Clustering Algorithm" << endl;

  ESHandle<L1CaloTriggerSetup> setup;
  iSetup.get<L1CaloTriggerSetupRcd>().get(setup);
  card = CaloClusteringCard(*setup);

   //Get ECAL + HCAL Digits from the EVent
   edm::Handle<L1CaloTowerCollection> towers;
   iEvent.getByLabel(towers_,towers);

   //Book the Collection
   std::auto_ptr<L1CaloClusterCollection> clusters (new L1CaloClusterCollection);
   card.clusterize(*clusters,*towers);


   //Put Clusters to file
   edm::LogInfo("ClusterProducer") << "Saving " << clusters->size() <<" Clusters"<< endl;
   iEvent.put(clusters);
}

// ------------ method called once each job just before starting event loop  ------------
void
L1CaloClusterProducer::beginJob(const edm::EventSetup& setup)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1CaloClusterProducer::endJob() {
}



