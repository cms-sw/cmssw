/* L1CaloTowerProducer
Reads TPGs, fixes the energy scale compression and produces towers
M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

//Includes for the Calo Scales
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "FWCore/Framework/interface/EventSetup.h"
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"


class L1CaloTowerProducer : public edm::EDProducer {
   public:
      explicit L1CaloTowerProducer(const edm::ParameterSet&);
      ~L1CaloTowerProducer();

   private:

      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      /*INPUTS*/

      //Calorimeter Digis
      edm::InputTag ecalDigis_;
      edm::InputTag hcalDigis_;

};




L1CaloTowerProducer::L1CaloTowerProducer(const edm::ParameterSet& iConfig):
  ecalDigis_(iConfig.getParameter<edm::InputTag>("ECALDigis")),
  hcalDigis_(iConfig.getParameter<edm::InputTag>("HCALDigis"))
{
  //Register Product
  produces<l1slhc::L1CaloTowerCollection>();
}


L1CaloTowerProducer::~L1CaloTowerProducer()
{

}


void
L1CaloTowerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace l1slhc;

   //Get ECAL + HCAL Digits from the EVent

   edm::Handle<EcalTrigPrimDigiCollection> ecalDigi;
   edm::Handle<HcalTrigPrimDigiCollection> hcalDigi;

   //Setup Calo Scales
   edm::ESHandle<L1CaloEcalScale> ecalScale;
   iSetup.get<L1CaloEcalScaleRcd>().get(ecalScale);
   const L1CaloEcalScale *eScale = ecalScale.product();

   edm::ESHandle<L1CaloHcalScale> hcalScale;
   iSetup.get<L1CaloHcalScaleRcd>().get(hcalScale);
   const L1CaloHcalScale *hScale = hcalScale.product();

   //get Tower Thresholds
   ESHandle<L1CaloTriggerSetup> s;
   iSetup.get<L1CaloTriggerSetupRcd>().get(s);
   const L1CaloTriggerSetup* setup= s.product(); 

   //Book the Collection
   std::auto_ptr<L1CaloTowerCollection> towers (new L1CaloTowerCollection);


   //Loop through the TPGs
   iEvent.getByLabel(ecalDigis_,ecalDigi);
   
   for(EcalTrigPrimDigiCollection::const_iterator ecalTower = ecalDigi->begin();ecalTower!=ecalDigi->end();++ecalTower)
     if(ecalTower->compressedEt()>0)
       {
	 L1CaloTower c;
	 c.setPos(ecalTower->id().ieta(),ecalTower->id().iphi());
	 int et = (int)(2*eScale->et(ecalTower->compressedEt(),abs(ecalTower->id().ieta()),ecalTower->id().ieta()/abs(ecalTower->id().ieta()) ));
	 c.setParams(et,0,ecalTower->fineGrain());
	 if(et>setup->ecalActivityThr())
	   towers->push_back(c);
       }
   


   //Now take the ecal towers and for common HCAL Towers fill them or add additional towers.....
   L1CaloTowerCollection additional;

   iEvent.getByLabel(hcalDigis_,hcalDigi);


   for(HcalTrigPrimDigiCollection::const_iterator hcalTower = hcalDigi->begin();hcalTower!=hcalDigi->end();++hcalTower)
     if(hcalTower->SOI_compressedEt()>0)
       {
	 
	 bool is_also_ecal = false;
	 for(L1CaloTowerCollection::iterator c=  towers->begin();c!=towers->end();++c)
	   {
	     if(hcalTower->id().ieta() == c->iEta() && hcalTower->id().iphi() == c->iPhi())
	       {
		 int et =(int)( 2*hScale->et(hcalTower->SOI_compressedEt(),abs(hcalTower->id().ieta()),hcalTower->id().ieta()/abs(hcalTower->id().ieta()) ));
		 if(et>setup->hcalActivityThr()) {
		   c->setParams(c->E(),et,c->fineGrain());
		   is_also_ecal = true;
		 }
	       }
	   }
	 //if there is no ecal tower add it additionall
	 if(!is_also_ecal)
	   {
	     L1CaloTower c;
	     c.setPos(hcalTower->id().ieta(),hcalTower->id().iphi());
	     int et =(int)(2*hScale->et(hcalTower->SOI_compressedEt(),abs(hcalTower->id().ieta()),hcalTower->id().ieta()/abs(hcalTower->id().ieta()) ));
	     c.setParams(0,et,false);
	     if(et>setup->hcalActivityThr())
	       additional.push_back(c);

	   }


       }

   
   towers->insert(towers->end(),additional.begin(),additional.end());
   iEvent.put(towers);
}


// ------------ method called once each job just after ending the event loop  ------------
void
L1CaloTowerProducer::endJob() {

}
DEFINE_ANOTHER_FWK_MODULE(L1CaloTowerProducer);
