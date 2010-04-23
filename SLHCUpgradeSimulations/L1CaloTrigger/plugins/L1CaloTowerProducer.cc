#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloTowerProducer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

//Includes for the Cal Scales

#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "FWCore/Framework/interface/EventSetup.h"

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


   //Book the Collection
   std::auto_ptr<L1CaloTowerCollection> towers (new L1CaloTowerCollection);


   //Loop through the TPGs
   if(iEvent.getByLabel(ecalDigis_,ecalDigi))
      for(EcalTrigPrimDigiCollection::const_iterator ecalTower = ecalDigi->begin();ecalTower!=ecalDigi->end();++ecalTower)
  if(ecalTower->compressedEt()>0)
    {
          L1CaloTower c;
          c.setPos(ecalTower->id().ieta(),ecalTower->id().iphi());
          int et = (int)(2*eScale->et(ecalTower->compressedEt(),abs(ecalTower->id().ieta()),ecalTower->id().ieta()/abs(ecalTower->id().ieta()) ));
          c.setParams(et,0,ecalTower->fineGrain());
          if(ecalTower->compressedEt() >0)
      towers->push_back(c);
    }


   //Now take the ecal towers and for common HCAL Towers fill them or add additional towers.....
   L1CaloTowerCollection additional;


  if(iEvent.getByLabel(hcalDigis_,hcalDigi))
      for(HcalTrigPrimDigiCollection::const_iterator hcalTower = hcalDigi->begin();hcalTower!=hcalDigi->end();++hcalTower)
        if(hcalTower->SOI_compressedEt()>0)
  {

    bool is_also_ecal = false;
    for(L1CaloTowerCollection::iterator c=  towers->begin();c!=towers->end();++c)
      {
        if(hcalTower->id().ieta() == c->iEta() && hcalTower->id().iphi() == c->iPhi())
    {
       int et =(int)( 2*hScale->et(hcalTower->SOI_compressedEt(),abs(hcalTower->id().ieta()),hcalTower->id().ieta()/abs(hcalTower->id().ieta()) ));
      c->setParams(c->E(),et,c->fineGrain());
      is_also_ecal = true;
    }
      }
    //if there is no ecal tower add it additionall
    if(!is_also_ecal)
      {
              L1CaloTower c;
        c.setPos(hcalTower->id().ieta(),hcalTower->id().iphi());
               int et =(int)(2*hScale->et(hcalTower->SOI_compressedEt(),abs(hcalTower->id().ieta()),hcalTower->id().ieta()/abs(hcalTower->id().ieta()) ));
        c.setParams(0,et,false);
        if(et>0)
        additional.push_back(c);

      }


  }


  towers->insert(towers->end(),additional.begin(),additional.end());
   iEvent.put(towers);






}

// ------------ method called once each job just before starting event loop  ------------
void
L1CaloTowerProducer::beginJob(const edm::EventSetup&)
{


}

// ------------ method called once each job just after ending the event loop  ------------
void
L1CaloTowerProducer::endJob() {

}



