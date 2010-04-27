/* L1CaloJetProducer
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
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/JetModule.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

class L1CaloJetProducer : public edm::EDProducer {
   public:
      explicit L1CaloJetProducer(const edm::ParameterSet&);
      ~L1CaloJetProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      /*INPUTS*/

      //Calorimeter Digis
      edm::InputTag regions_;
      bool verbosity_;
      JetModule clusteringModule;
};


   using namespace edm;
   using namespace std;

   using namespace l1slhc;


L1CaloJetProducer::L1CaloJetProducer(const edm::ParameterSet& iConfig):
  regions_(iConfig.getParameter<edm::InputTag>("src")),
  verbosity_(iConfig.getUntrackedParameter<bool>("verbosity",false))
{
  //Register Product
  produces<l1slhc::L1CaloJetCollection>();
}


L1CaloJetProducer::~L1CaloJetProducer()
{

}



void
L1CaloJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  ESHandle<L1CaloTriggerSetup> setup;
  iSetup.get<L1CaloTriggerSetupRcd>().get(setup);

  clusteringModule = JetModule(*setup);
  
  //Get ECAL + HCAL Digits from the EVent
  edm::Handle<L1CaloRegionCollection> regions;
  iEvent.getByLabel(regions_,regions);
  
  //Book the Collection
   std::auto_ptr<L1CaloJetCollection> jets (new L1CaloJetCollection);

   clusteringModule.clusterize(*jets,regions);

   if(verbosity_) {
     printf("RawJets---------------\n\n");
     for(unsigned int i=0;i<jets->size();++i)
       std::cout << jets->at(i)<<std::endl;

   }

   iEvent.put(jets);
}



DEFINE_ANOTHER_FWK_MODULE(L1CaloJetProducer);
