/* L1CaloTowerProducer Reads TPGs, fixes the energy scale compression and
   produces towers

   M.Bachtis,S.Dasu University of Wisconsin-Madison

   Modified Andrew W. Rose Imperial College, London */

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

// Includes for the Calo Scales
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

#include <map>
#include <deque>


class L1CaloTowerProducer:public edm::EDProducer
{
  public:
	explicit L1CaloTowerProducer( const edm::ParameterSet & );
	 ~L1CaloTowerProducer(  );

  private:

	virtual void produce( edm::Event &, const edm::EventSetup & );
	virtual void endJob(  );

	void addHcal( const int &, const int &, const int &, const bool & );
	void addEcal( const int &, const int &, const int &, const bool & );

	std::auto_ptr < l1slhc::L1CaloTowerCollection > mCaloTowers;

	const L1CaloTriggerSetup *mCaloTriggerSetup;
	const L1CaloEcalScale *mEcalScale;
	const L1CaloHcalScale *mHcalScale;

	// Calorimeter Digis
	edm::InputTag mEcalDigiInputTag;
	edm::InputTag mHcalDigiInputTag;

	bool mUseupgradehcal;
};




L1CaloTowerProducer::L1CaloTowerProducer( const edm::ParameterSet & aConfig ):
  mCaloTowers( NULL ),
  mEcalDigiInputTag( aConfig.getParameter < edm::InputTag > ( "ECALDigis" ) ),
  mHcalDigiInputTag( aConfig.getParameter < edm::InputTag > ( "HCALDigis" ) ),
  mUseupgradehcal( aConfig.getParameter < bool > ( "UseUpgradeHCAL" ) )
{
  // Register Product
  produces < l1slhc::L1CaloTowerCollection > (  );
}


L1CaloTowerProducer::~L1CaloTowerProducer(  )
{

}

void L1CaloTowerProducer::addHcal( const int &aCompressedEt, const int &aIeta,
								   const int &aIphi, const bool & aFG )
{
	if ( aCompressedEt > 0 )
	{
		int lET = ( int )( 2 * mHcalScale->et( aCompressedEt,
											   abs( aIeta ),
											   ( aIeta > 0 ? +1 : -1 ) ) );

		l1slhc::L1CaloTowerCollection::iterator lItr = mCaloTowers -> find ( aIeta, aIphi  );

		if ( lItr != ( *mCaloTowers ).end(  ) )
		{
			if ( lET > mCaloTriggerSetup->hcalActivityThr(  ) )
				lItr->setHcal( lET, aFG );
		}
		else
		{
			l1slhc::L1CaloTower lCaloTower( aIeta, aIphi );
			lCaloTower.setHcal( lET, aFG );
			if ( lET > mCaloTriggerSetup->hcalActivityThr(  ) )
				mCaloTowers->insert( aIeta , aIphi , lCaloTower );
		}
	}
}

void L1CaloTowerProducer::addEcal( const int &aCompressedEt, const int &aIeta,
								   const int &aIphi, const bool & aFG )
{
	if ( aCompressedEt > 0 )
	{
		int lET = ( int )( 2 * mEcalScale->et( aCompressedEt,
											   abs( aIeta ),
											   ( aIeta > 0 ? +1 : -1 ) ) );

		l1slhc::L1CaloTower lCaloTower( aIeta, aIphi );
		lCaloTower.setEcal( lET, aFG );

		if ( lET > mCaloTriggerSetup->ecalActivityThr(  ) )
			mCaloTowers->insert( aIeta , aIphi , lCaloTower );

	}
}

void L1CaloTowerProducer::produce( edm::Event & aEvent,
								   const edm::EventSetup & aSetup )
{

	// create a new l1slhc::L1CaloTowerCollection (auto_ptr should handle deletion of the last one correctly)
	mCaloTowers = std::auto_ptr < l1slhc::L1CaloTowerCollection > ( new l1slhc::L1CaloTowerCollection );

	// Setup Calo Scales
	edm::ESHandle < L1CaloEcalScale > lEcalScaleHandle;
	aSetup.get < L1CaloEcalScaleRcd > (  ).get( lEcalScaleHandle );
	mEcalScale = lEcalScaleHandle.product(  );

	edm::ESHandle < L1CaloHcalScale > lHcalScaleHandle;
	aSetup.get < L1CaloHcalScaleRcd > (  ).get( lHcalScaleHandle );
	mHcalScale = lHcalScaleHandle.product(  );

	// get Tower Thresholds
	edm::ESHandle < L1CaloTriggerSetup > mCaloTriggerSetupHandle;
	aSetup.get < L1CaloTriggerSetupRcd > (  ).get( mCaloTriggerSetupHandle );
	mCaloTriggerSetup = mCaloTriggerSetupHandle.product(  );


	// Loop through the TPGs
	//getting data from event takes 3 orders of magnitude longer than anything else in the program : O(10-100ms) cf O(10-100us)
	edm::Handle < EcalTrigPrimDigiCollection > lEcalDigiHandle;
	aEvent.getByLabel( mEcalDigiInputTag, lEcalDigiHandle );

	for ( EcalTrigPrimDigiCollection::const_iterator lEcalTPItr = lEcalDigiHandle->begin(  ); lEcalTPItr != lEcalDigiHandle->end(  ); ++lEcalTPItr )
		addEcal( lEcalTPItr->compressedEt(  ), lEcalTPItr->id(  ).ieta(  ), lEcalTPItr->id(  ).iphi(  ), lEcalTPItr->fineGrain(  ) );

	if ( !mUseupgradehcal )
	{
		//getting data from event takes 3 orders of magnitude longer than anything else in the program : O(10-100ms) cf O(10-100us)
		edm::Handle < HcalTrigPrimDigiCollection > lHcalDigiHandle;
		aEvent.getByLabel( mHcalDigiInputTag, lHcalDigiHandle );

		for ( HcalTrigPrimDigiCollection::const_iterator lHcalTPItr = lHcalDigiHandle->begin(  ); lHcalTPItr != lHcalDigiHandle->end(  ); ++lHcalTPItr )
			addHcal( lHcalTPItr->SOI_compressedEt(  ), lHcalTPItr->id(  ).ieta(  ), lHcalTPItr->id(  ).iphi(  ), lHcalTPItr->SOI_fineGrain(  ) );
	}
	else
	{
          // Detect if the upgrade HCAL header file is included
#ifdef DIGIHCAL_HCALUPGRADETRIGGERPRIMITIVEDIGI_H
#warning Not really a warning: just letting you know that Im enabling upgrade HCAL digis
		//getting data from event takes 3 orders of magnitude longer than anything else in the program : O(10-100ms) cf O(10-100us)
		edm::Handle < HcalUpgradeTrigPrimDigiCollection > lHcalDigiHandle;
		aEvent.getByLabel( mHcalDigiInputTag, lHcalDigiHandle );

		for ( HcalUpgradeTrigPrimDigiCollection::const_iterator lHcalTPItr = lHcalDigiHandle->begin(  ); lHcalTPItr != lHcalDigiHandle->end(  ); ++lHcalTPItr )
			addHcal( lHcalTPItr->SOI_compressedEt(  ), lHcalTPItr->id(  ).ieta(  ), lHcalTPItr->id(  ).iphi(  ), lHcalTPItr->SOI_fineGrain(  ) );
#else
#warning Not really a warning: just letting you know that Im NOT enabling upgrade HCAL digis
                // If the user tries to specify this option, but it isn't
                // available, throw an exception.
                throw cms::Exception("NotImplmented") <<
                  "You requested to use the upgrade HCAL digis.  However the "
                  << "L1CaloTowerProducer.cc module was not compiled with "
                  << "support for them.  "
                  << "Please edit SLHCUpSims/L1CaloTrig/plugins/L1CaloTowerProducer.cc" << std::endl;
#endif
	}
	aEvent.put( mCaloTowers );
}


// ------------ method called once each job just after ending the event loop
// ------------
void L1CaloTowerProducer::endJob(  )
{

}


DEFINE_EDM_PLUGIN( edm::MakerPluginFactory,
				   edm::WorkerMaker < L1CaloTowerProducer >,
				   "L1CaloTowerProducer" );
DEFINE_FWK_PSET_DESC_FILLER( L1CaloTowerProducer );
