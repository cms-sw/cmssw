/* L1CaloClusterProducer
Performs Clustering in Calorimeter
and produces a Cluster Collection

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
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/ClusteringModule.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

class L1CaloClusterProducer : public edm::EDProducer {
   public:
      explicit L1CaloClusterProducer(const edm::ParameterSet&);
      ~L1CaloClusterProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      /*INPUTS*/

      //Calorimeter Digis
      edm::InputTag towers_;
      bool verbosity_;
      ClusteringModule module;
};


   using namespace edm;
   using namespace std;

   using namespace l1slhc;


L1CaloClusterProducer::L1CaloClusterProducer(const edm::ParameterSet& iConfig):
  towers_(iConfig.getParameter<edm::InputTag>("src")),
  verbosity_(iConfig.getUntrackedParameter<bool>("verbosity",false))
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
  module = ClusteringModule(*setup);
  
  //Get ECAL + HCAL Digits from the EVent
  edm::Handle<L1CaloTowerCollection> towers;
  iEvent.getByLabel(towers_,towers);
  
  //Book the Collection
   std::auto_ptr<L1CaloClusterCollection> clusters (new L1CaloClusterCollection);
   module.clusterize(*clusters,towers);
   

   //Put Clusters to file
   edm::LogInfo("ClusterProducer") << "Saving " << clusters->size() <<" Clusters"<< endl;

   if(verbosity_) {
     printf("RAW Clusters---------------\n\n");
     for(unsigned int i=0;i<clusters->size();++i)
       std::cout << clusters->at(i)<<std::endl;
   }


   iEvent.put(clusters);
}



DEFINE_ANOTHER_FWK_MODULE(L1CaloClusterProducer);
