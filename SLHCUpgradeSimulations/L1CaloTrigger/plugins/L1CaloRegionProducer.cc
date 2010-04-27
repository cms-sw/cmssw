/* L1CaloRegionProducer
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
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/RegionModule.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

class L1CaloRegionProducer : public edm::EDProducer {
   public:
      explicit L1CaloRegionProducer(const edm::ParameterSet&);
      ~L1CaloRegionProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      /*INPUTS*/

      //Calorimeter Digis
      edm::InputTag towers_;
      RegionModule module;
};


   using namespace edm;
   using namespace std;

   using namespace l1slhc;


L1CaloRegionProducer::L1CaloRegionProducer(const edm::ParameterSet& iConfig):
  towers_(iConfig.getParameter<edm::InputTag>("src"))
{
  //Register Product
  produces<l1slhc::L1CaloRegionCollection>();
}


L1CaloRegionProducer::~L1CaloRegionProducer()
{

}



void
L1CaloRegionProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{



  ESHandle<L1CaloTriggerSetup> setup;
  iSetup.get<L1CaloTriggerSetupRcd>().get(setup);
  module = RegionModule(*setup);
  
  //Get ECAL + HCAL Digits from the EVent
  edm::Handle<L1CaloTowerCollection> towers;
  iEvent.getByLabel(towers_,towers);
  
  //Book the Collection
   std::auto_ptr<L1CaloRegionCollection> regions (new L1CaloRegionCollection);
   module.clusterize(*regions,towers);


   //Put Clusters to file
   iEvent.put(regions);
}
DEFINE_ANOTHER_FWK_MODULE(L1CaloRegionProducer);
