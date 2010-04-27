/* L1CaloClusterFilter
Removes Cluster Overlap
and applies isolation criteria
M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/
// system include files
#include <memory>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/FilteringModule.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"

class L1CaloClusterFilter : public edm::EDProducer {
   public:
      explicit L1CaloClusterFilter(const edm::ParameterSet&);
      ~L1CaloClusterFilter();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      /*INPUTS*/

      //Calorimeter Digis
      edm::InputTag clusters_;
      bool verbosity_;
      FilteringModule filterModule;
};


using namespace edm;
using namespace std;
using namespace l1slhc;


L1CaloClusterFilter::L1CaloClusterFilter(const edm::ParameterSet& iConfig):
  clusters_(iConfig.getParameter<edm::InputTag>("src")),
  verbosity_(iConfig.getUntrackedParameter<bool>("verbosity",false))
{
  //Register Product
  produces<l1slhc::L1CaloClusterCollection>();
}


L1CaloClusterFilter::~L1CaloClusterFilter()
{

}



void
L1CaloClusterFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace l1slhc;
  edm::LogInfo ("INFO") << "Starting Cluster Filtering Algorithm" << endl;

  ESHandle<L1CaloTriggerSetup> setup;
  iSetup.get<L1CaloTriggerSetupRcd>().get(setup);

  filterModule = FilteringModule(*setup);
   //Get ECAL + HCAL Digits from the EVent
   edm::Handle<L1CaloClusterCollection> clusters;
   iEvent.getByLabel(clusters_,clusters);
   //Book the Collection to be saved
   std::auto_ptr<L1CaloClusterCollection> filteredClusters (new L1CaloClusterCollection);
   //Apply the algorithms
   filterModule.cleanClusters(clusters,*filteredClusters);
   //Put Clusters to file
   std::sort(filteredClusters->begin(),filteredClusters->end(),HigherClusterEt());

   if(verbosity_) {
     printf("Filtered Clusters---------------\n\n");
     for(unsigned int i=0;i<filteredClusters->size();++i)
       std::cout << filteredClusters->at(i)<<std::endl;

   }

   iEvent.put(filteredClusters);
}

DEFINE_ANOTHER_FWK_MODULE(L1CaloClusterFilter);
