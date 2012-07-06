/* L1RingSubtractionProducer Reads TPGs, fixes the energy scale compression and
   produces towers

   Andrew W. Rose Imperial College, London */

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


#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"

#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"

#include <algorithm> 
#include <string> 


class L1RingSubtractionProducer:public edm::EDProducer
{
  public:
	explicit L1RingSubtractionProducer( const edm::ParameterSet & );
	 ~L1RingSubtractionProducer(  );

  private:

	virtual void produce( edm::Event &, const edm::EventSetup & );
	virtual void endJob(  );

	edm::ESHandle < L1CaloTriggerSetup >  mCaloTriggerSetup;

	edm::InputTag mInputCollectionTag;
	edm::Handle < l1slhc::L1CaloTowerCollection > mInputCollection;
	std::auto_ptr < l1slhc::L1CaloTowerCollection > mOutputCollection;


	double getEcalConstant( int& iEta  );
	double getHcalConstant( int& iEta );


	enum{
		constant,
		mean,
		median
	}
	mRingSubtractionType;


};




L1RingSubtractionProducer::L1RingSubtractionProducer( const edm::ParameterSet & aConfig ):
mInputCollectionTag( aConfig.getParameter < edm::InputTag > ( "src" ) ),
mOutputCollection( NULL )
{
	
	std::string lRingSubtractionType = aConfig.getParameter< std::string >("RingSubtractionType");
	std::transform( lRingSubtractionType.begin() , lRingSubtractionType.end() , lRingSubtractionType.begin() , ::toupper ); //do the comparison in upper case so config file can read "Mean", "mean", "MEAN", "mEaN", etc. and give the same result.

	if( lRingSubtractionType == "MEAN" ){
		mRingSubtractionType = mean;
	}else if( lRingSubtractionType == "MEDIAN" ){ 
		mRingSubtractionType = median;
	}else{
		mRingSubtractionType = constant;
	}

	// Register Product
	produces < l1slhc::L1CaloTowerCollection > (  );
}


L1RingSubtractionProducer::~L1RingSubtractionProducer(  )
{}





void L1RingSubtractionProducer::produce( edm::Event & aEvent,
								   const edm::EventSetup & aSetup )
{

	if( mRingSubtractionType == constant ){
		std::cout << "!!! WARNING !!! Constant ring subtraction is yet not implemented. A constant of 0 will be assumed !!! WARNING !!!\n" << std::endl;
	}

	aSetup.get < L1CaloTriggerSetupRcd > (  ).get( mCaloTriggerSetup );

	aEvent.getByLabel( mInputCollectionTag, mInputCollection );

	// create a new l1slhc::L1CaloTowerCollection (auto_ptr should handle deletion of the last one correctly)
	mOutputCollection = std::auto_ptr < l1slhc::L1CaloTowerCollection > ( new l1slhc::L1CaloTowerCollection );

	if( mRingSubtractionType  == mean ){

		std::map< int , double > lMeanEcal , lMeanHcal;

		for( l1slhc::L1CaloTowerCollection::const_iterator lInputIt = mInputCollection->begin() ; lInputIt != mInputCollection->end() ; ++lInputIt ){
			lMeanEcal[ lInputIt->iEta() ] += double(lInputIt->E());
			lMeanHcal[ lInputIt->iEta() ] += double(lInputIt->H());
		}

		//Empty towers are assumed to have zero energy contribution, so we divide by all towers
		for( std::map< int , double >::iterator lIt = lMeanEcal.begin() ; lIt != lMeanEcal.end() ; ++lIt ){
			lIt->second /= 72;
		}

		//Empty towers are assumed to have zero energy contribution, so we divide by all towers
		for( std::map< int , double >::iterator lIt = lMeanHcal.begin() ; lIt != lMeanHcal.end() ; ++lIt ){
			lIt->second /= 72;
		}

		for( l1slhc::L1CaloTowerCollection::const_iterator lInputIt = mInputCollection->begin() ; lInputIt != mInputCollection->end() ; ++lInputIt ){
			int lEta = 	lInputIt->iEta();		
			int lPhi = 	lInputIt->iPhi();		

			int lEcal = int( double(lInputIt->E()) - lMeanEcal[ lEta ] );
			int lHcal = int( double(lInputIt->H()) - lMeanHcal[ lEta ] );

			l1slhc::L1CaloTower lCaloTower( lEta , lPhi );
			lCaloTower.setEcal( lEcal , lInputIt->EcalFG() );
			lCaloTower.setHcal( lHcal , lInputIt->HcalFG() );
	
			mOutputCollection->insert( lEta , lPhi , lCaloTower );
		}


	} else if( mRingSubtractionType  == median ){

		std::map< int , std::deque<int> > lEcals , lHcals;

		for( l1slhc::L1CaloTowerCollection::const_iterator lInputIt = mInputCollection->begin() ; lInputIt != mInputCollection->end() ; ++lInputIt ){
			lEcals[ lInputIt->iEta() ].push_back( lInputIt->E() );
			lHcals[ lInputIt->iEta() ].push_back( lInputIt->H() );

			//std::cout<<"ECal energy: "<<lInputIt->E()<<std::endl;
			//std::cout<<"HCal energy: "<<lInputIt->H()<<std::endl;
		}

		std::map< int , double > lMedianEcal , lMedianHcal;

		//Empty towers are assumed to have zero energy contribution
		for( std::map< int , std::deque<int> >::iterator lIt = lEcals.begin() ; lIt != lEcals.end() ; ++lIt ){
			lIt->second.resize( 72 , 0 );
			std::sort( lIt->second.begin() , lIt->second.end() );
			lMedianEcal[ lIt->first ] = (lIt->second.at( 35 ) + lIt->second.at( 36 )) / 2.0;		
	}


		//Empty towers are assumed to have zero energy contribution
		for( std::map< int , std::deque<int> >::iterator lIt = lHcals.begin() ; lIt != lHcals.end() ; ++lIt ){
			lIt->second.resize( 72 , 0 );
			std::sort( lIt->second.begin() , lIt->second.end() );
			lMedianHcal[ lIt->first ] = (lIt->second.at( 35 ) + lIt->second.at( 36 )) / 2.0;
	
	}


		for( l1slhc::L1CaloTowerCollection::const_iterator lInputIt = mInputCollection->begin() ; lInputIt != mInputCollection->end() ; ++lInputIt ){
			int lEta = 	lInputIt->iEta();		
			int lPhi = 	lInputIt->iPhi();		

			int lEcal = int( double(lInputIt->E()) - lMedianEcal[ lEta ] );
			int lHcal = int( double(lInputIt->H()) - lMedianHcal[ lEta ] );

			l1slhc::L1CaloTower lCaloTower( lEta , lPhi );
			lCaloTower.setEcal( lEcal , lInputIt->EcalFG() );
			lCaloTower.setHcal( lHcal , lInputIt->HcalFG() );
	
			mOutputCollection->insert( lEta , lPhi , lCaloTower );
		}

	}else{


		for( l1slhc::L1CaloTowerCollection::const_iterator lInputIt = mInputCollection->begin() ; lInputIt != mInputCollection->end() ; ++lInputIt ){
			int lEta = 	lInputIt->iEta();		
			int lPhi = 	lInputIt->iPhi();		

			int lEcal = int( double(lInputIt->E()) - getEcalConstant( lEta ) );
			int lHcal = int( double(lInputIt->H()) - getHcalConstant( lEta ) );

			l1slhc::L1CaloTower lCaloTower( lEta , lPhi );
			lCaloTower.setEcal( lEcal , lInputIt->EcalFG() );
			lCaloTower.setHcal( lHcal , lInputIt->HcalFG() );
	
			mOutputCollection->insert( lEta , lPhi , lCaloTower );
		}


	}
	
	
	
	

	
	aEvent.put( mOutputCollection );
}











double L1RingSubtractionProducer::getEcalConstant( int& iEta ){
	//can use mCaloTriggerSetup member object to retrieve a look-up table from EventSetup
	//std::cout << "Function " << __PRETTY_FUNCTION__ << " not implemented. Returning 0 for subtraction." << std::endl;
	return 0;
}


double L1RingSubtractionProducer::getHcalConstant( int& iEta ){
	//can use mCaloTriggerSetup member object to retrieve a look-up table from EventSetup
	//std::cout << "Function " << __PRETTY_FUNCTION__ << " not implemented. Returning 0 for subtraction." << std::endl;
	return 0;
}




// ------------ method called once each job just after ending the event loop
// ------------
void L1RingSubtractionProducer::endJob(  )
{
}


DEFINE_EDM_PLUGIN( edm::MakerPluginFactory,
				   edm::WorkerMaker < L1RingSubtractionProducer >,
				   "L1RingSubtractionProducer" );
DEFINE_FWK_PSET_DESC_FILLER( L1RingSubtractionProducer );
