#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1JetMaker.h"

   using namespace edm;
   using namespace std;

   using namespace l1slhc;


L1JetMaker::L1JetMaker(const edm::ParameterSet& iConfig):
  clusters_(iConfig.getParameter<edm::InputTag>("FilteredClusters"))
{
  //Register Product
  produces<l1slhc::L1CaloJetCollection>();
}


L1JetMaker::~L1JetMaker()
{

}



void
L1JetMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  ESHandle<L1CaloTriggerSetup> setup;
  iSetup.get<L1CaloTriggerSetupRcd>().get(setup);
  card = CaloJetCard(*setup);

   //Get ECAL + HCAL Digits from the EVent
   edm::Handle<L1CaloClusterCollection> clusters;
   iEvent.getByLabel(clusters_,clusters);

   //Book the Collection
   std::auto_ptr<L1CaloJetCollection> jets (new L1CaloJetCollection);
   card.makeJets(*clusters,*jets);


   //Put Clusters to file
   edm::LogInfo("ClusterProducer") << "Saving " << jets->size() <<" Jets"<< endl;
   iEvent.put(jets);
}

// ------------ method called once each job just before starting event loop  ------------
void
L1JetMaker::beginJob(const edm::EventSetup& setup)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1JetMaker::endJob() {
}



