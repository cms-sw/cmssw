/* L1CaloTowerProducer
Reads TPGs, fixes the energy scale compression and produces towers
M.Bachtis,S.Dasu
University of Wisconsin-Madison

Modified Andrew W. Rose
Imperial College, London
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

//added to allow use of Upgrade HCAL - AWR 12/05/2011
#include "DataFormats/HcalDigi/interface/HcalUpgradeTriggerPrimitiveDigi.h"

#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"

#include <map>
#include <deque>


class L1CaloTowerProducer : public edm::EDProducer {
   public:
      explicit L1CaloTowerProducer(const edm::ParameterSet&);
      ~L1CaloTowerProducer();

   private:

      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;


      /*INPUTS*/
      void addHcal( const int& , const int& , const int& , const bool& );
      void addEcal( const int& , const int& , const int& , const bool& );

      typedef std::map< std::pair<int,int> , l1slhc::L1CaloTower > tAssociationMap;
      tAssociationMap mAssociationMap;
 
      const L1CaloTriggerSetup* mCaloTriggerSetup;
      const L1CaloEcalScale* mEcalScale;
      const L1CaloHcalScale* mHcalScale;

      //Calorimeter Digis
      edm::InputTag mEcalDigiInputTag;
      edm::InputTag mHcalDigiInputTag;
 
//added to allow use of Upgrade HCAL - AWR 12/05/2011
     bool useUpgradeHCAL_;

};




L1CaloTowerProducer::L1CaloTowerProducer(const edm::ParameterSet& aConfig):
  mEcalDigiInputTag(aConfig.getParameter<edm::InputTag>("ECALDigis")),
  mHcalDigiInputTag(aConfig.getParameter<edm::InputTag>("HCALDigis")),
  useUpgradeHCAL_(aConfig.getParameter<bool>("UseUpgradeHCAL")) //added to allow use of Upgrade HCAL - AWR 12/05/2011
{
  //Register Product
  produces<l1slhc::L1CaloTowerCollection>();
}


L1CaloTowerProducer::~L1CaloTowerProducer()
{

}



void
L1CaloTowerProducer::addHcal( const int& aCompressedEt, const int& aIeta, const int& aIphi , const bool& aFG  ){
   if(aCompressedEt>0)
       {
	 int lET =(int)( 2*
		mHcalScale->et(
			aCompressedEt,
			abs(aIeta),
			(aIeta>0?+1:-1)
		)
	 );
	 
	 tAssociationMap::iterator lItr = mAssociationMap.find( std::make_pair(aIeta,aIphi) );

	 if ( lItr != mAssociationMap.end() ){
		 if( lET > mCaloTriggerSetup->hcalActivityThr() ) lItr->second.setHcal( lET , aFG );
	 }else{
		 l1slhc::L1CaloTower lCaloTower;
		 lCaloTower.setPos(aIeta,aIphi);
		 lCaloTower.setHcal( lET , aFG );
	    	 if( lET > mCaloTriggerSetup->hcalActivityThr() ) mAssociationMap[ std::make_pair(aIeta,aIphi) ] = lCaloTower;
	 }
     }
}


void
L1CaloTowerProducer::addEcal( const int& aCompressedEt, const int& aIeta, const int& aIphi , const bool& aFG ){
    if(aCompressedEt>0){
   	 int lET = (int)(2*
			mEcalScale->et(
				aCompressedEt,
				abs(aIeta),
				(aIeta>0?+1:-1)
			)
		 );

	 l1slhc::L1CaloTower lCaloTower;
	 lCaloTower.setPos( aIeta , aIphi );
	 lCaloTower.setEcal( lET , aFG );

	 if( lET > mCaloTriggerSetup->ecalActivityThr() ) mAssociationMap[ std::make_pair(aIeta,aIphi) ] = lCaloTower;
    }
}






void
L1CaloTowerProducer::produce(edm::Event& aEvent, const edm::EventSetup& aSetup)
{
   using namespace edm;
   using namespace l1slhc;
  
   //clear the container
   mAssociationMap.clear();
 

   //Setup Calo Scales
   edm::ESHandle<L1CaloEcalScale> lEcalScaleHandle;
   aSetup.get<L1CaloEcalScaleRcd>().get(lEcalScaleHandle);
   mEcalScale = lEcalScaleHandle.product();

   edm::ESHandle<L1CaloHcalScale> lHcalScaleHandle;
   aSetup.get<L1CaloHcalScaleRcd>().get(lHcalScaleHandle);
   mHcalScale = lHcalScaleHandle.product();

   //get Tower Thresholds
   ESHandle<L1CaloTriggerSetup> mCaloTriggerSetupHandle;
   aSetup.get<L1CaloTriggerSetupRcd>().get(mCaloTriggerSetupHandle);
   mCaloTriggerSetup = mCaloTriggerSetupHandle.product(); 



   //Loop through the TPGs
   edm::Handle<EcalTrigPrimDigiCollection> lEcalDigiHandle;
   aEvent.getByLabel(mEcalDigiInputTag,lEcalDigiHandle);
   
   for(EcalTrigPrimDigiCollection::const_iterator lEcalTPItr = lEcalDigiHandle->begin();lEcalTPItr!=lEcalDigiHandle->end();++lEcalTPItr)
	addEcal( lEcalTPItr->compressedEt() , lEcalTPItr->id().ieta() , lEcalTPItr->id().iphi() , lEcalTPItr->fineGrain() );
   


//added to allow use of Upgrade HCAL - AWR 12/05/2011
  if( ! useUpgradeHCAL_ ){
    edm::Handle<HcalTrigPrimDigiCollection> lHcalDigiHandle;
    aEvent.getByLabel(mHcalDigiInputTag,lHcalDigiHandle);

    for(HcalTrigPrimDigiCollection::const_iterator lHcalTPItr = lHcalDigiHandle->begin();lHcalTPItr!=lHcalDigiHandle->end();++lHcalTPItr)
      addHcal(lHcalTPItr->SOI_compressedEt(), lHcalTPItr->id().ieta(), lHcalTPItr->id().iphi() , lHcalTPItr->SOI_fineGrain() );
 
  }else{
    edm::Handle<HcalUpgradeTrigPrimDigiCollection> lHcalDigiHandle;
    aEvent.getByLabel(mHcalDigiInputTag,lHcalDigiHandle);

    for(HcalUpgradeTrigPrimDigiCollection::const_iterator lHcalTPItr = lHcalDigiHandle->begin();lHcalTPItr!=lHcalDigiHandle->end();++lHcalTPItr)
      addHcal(lHcalTPItr->SOI_compressedEt(), lHcalTPItr->id().ieta(), lHcalTPItr->id().iphi() , lHcalTPItr->SOI_fineGrain() );

  }
   

   //Book the Collection
   // could avoid this copy operation if we used a deque rather than a vector to store the towers and then just keep pointers as the second elements in the map.
   std::auto_ptr<L1CaloTowerCollection> lCaloTowersOut (new L1CaloTowerCollection);
   lCaloTowersOut->reserve( mAssociationMap.size() );

   for( tAssociationMap::iterator lItr=mAssociationMap.begin(); lItr!=mAssociationMap.end(); ++lItr){
	lCaloTowersOut->push_back( lItr->second );
   }

   aEvent.put(lCaloTowersOut);
}


// ------------ method called once each job just after ending the event loop  ------------
void
L1CaloTowerProducer::endJob() {

}
//#define DEFINE_ANOTHER_FWK_MODULE(type) DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type); DEFINE_FWK_PSET_DESC_FILLER(type)
DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<L1CaloTowerProducer>,"L1CaloTowerProducer"); DEFINE_FWK_PSET_DESC_FILLER(L1CaloTowerProducer);
//DEFINE_ANOTHER_FWK_MODULE(L1CaloTowerProducer);
