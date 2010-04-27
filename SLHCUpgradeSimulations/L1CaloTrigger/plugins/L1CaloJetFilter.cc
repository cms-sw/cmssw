/* L1CaloJetFilter
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
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/JetFilteringModule.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

class L1CaloJetFilter : public edm::EDProducer {
   public:
      explicit L1CaloJetFilter(const edm::ParameterSet&);
      ~L1CaloJetFilter();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      /*INPUTS*/

      //Calorimeter Digis
      edm::InputTag jets_;
      bool verbosity_;
      JetFilteringModule filteringModule;
};


   using namespace edm;
   using namespace std;

   using namespace l1slhc;


L1CaloJetFilter::L1CaloJetFilter(const edm::ParameterSet& iConfig):
  jets_(iConfig.getParameter<edm::InputTag>("src")),
  verbosity_(iConfig.getUntrackedParameter<bool>("verbosity",false))
{
  //Register Product
  produces<l1slhc::L1CaloJetCollection>();
}


L1CaloJetFilter::~L1CaloJetFilter()
{

}



void
L1CaloJetFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  ESHandle<L1CaloTriggerSetup> setup;
  iSetup.get<L1CaloTriggerSetupRcd>().get(setup);

  filteringModule = JetFilteringModule(*setup);
  
  //Get ECAL + HCAL Digits from the EVent
  edm::Handle<L1CaloJetCollection> ijets;
  iEvent.getByLabel(jets_,ijets);
  
  //Book the Collection
   std::auto_ptr<L1CaloJetCollection> jets (new L1CaloJetCollection);

   filteringModule.cleanClusters(ijets,*jets);

   if(verbosity_) {
     printf("Filtered Jets---------------\n\n");
     for(unsigned int i=0;i<jets->size();++i)
       std::cout << jets->at(i)<<std::endl;
   }


   iEvent.put(jets);
}

DEFINE_ANOTHER_FWK_MODULE(L1CaloJetFilter);
