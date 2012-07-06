#ifndef L1CaloAlgoBase_h
#define L1CaloAlgoBase_h

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
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"


#ifdef __GNUG__
#include <cxxabi.h>
#endif




template < typename tInputCollection ,  typename tOutputCollection >
class L1CaloAlgoBase: 
public edm::EDProducer
{
  public:
	explicit L1CaloAlgoBase( const edm::ParameterSet & );
	virtual ~L1CaloAlgoBase(  );

	virtual void initialize( ){}

	virtual void algorithm( const int &, const int & ) = 0;

	std::string sourceName() const;

  private:
	void produce( edm::Event &, const edm::EventSetup & );

	edm::InputTag mInputCollectionTag;

  protected:
	typename tInputCollection::const_iterator fetch( const int &, const int & );

	bool mVerbosity;

	int mPhiOffset, mEtaOffset, mPhiIncrement, mEtaIncrement;

	edm::ESHandle < L1CaloTriggerSetup >  mCaloTriggerSetup;

	edm::Handle < tInputCollection > mInputCollection;
	std::auto_ptr < tOutputCollection > mOutputCollection;
};





template < typename tInputCollection ,  typename tOutputCollection >
L1CaloAlgoBase< tInputCollection ,  tOutputCollection >::L1CaloAlgoBase ( const edm::ParameterSet & iConfig ):
mInputCollectionTag( iConfig.getParameter < edm::InputTag > ( "src" ) ),
mVerbosity( iConfig.getUntrackedParameter < bool > ( "verbosity", false ) ), 
mPhiOffset( 0 ), 
mEtaOffset( 0 ), 
mPhiIncrement( 1 ), 
mEtaIncrement( 1 )
{
	produces < tOutputCollection > (  );
}


template < typename tInputCollection ,  typename tOutputCollection >
L1CaloAlgoBase< tInputCollection ,  tOutputCollection >::~L1CaloAlgoBase(  )
{
}

template < typename tInputCollection ,  typename tOutputCollection >
std::string L1CaloAlgoBase< tInputCollection ,  tOutputCollection >::sourceName() const
{
	if( mInputCollectionTag.instance().size() )
	{
		return std::string( mInputCollectionTag.label () + " - " + mInputCollectionTag.instance() );
	}else{
		return std::string( mInputCollectionTag.label () );
	}
}

template < typename tInputCollection ,  typename tOutputCollection >
typename tInputCollection::const_iterator L1CaloAlgoBase< tInputCollection ,  tOutputCollection >::fetch( const int &aEta, const int &aPhi ){
	int lIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
	std::pair < int, int >lEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lIndex );
	return mInputCollection->find( lEtaPhi.first , lEtaPhi.second );
}







template < typename tInputCollection ,  typename tOutputCollection >
void L1CaloAlgoBase< tInputCollection ,  tOutputCollection >::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{

	iSetup.get < L1CaloTriggerSetupRcd > (  ).get( mCaloTriggerSetup );
	this->initialize();

	iEvent.getByLabel( mInputCollectionTag, mInputCollection );

	mOutputCollection = std::auto_ptr < tOutputCollection > ( new tOutputCollection );


	int lPhiMin = mCaloTriggerSetup->phiMin(  );
	int lPhiMax = mCaloTriggerSetup->phiMax(  ) + mPhiOffset;
	int lEtaMin = mCaloTriggerSetup->etaMin(  );
	int lEtaMax = mCaloTriggerSetup->etaMax(  ) + mEtaOffset;

	for ( int lEta = lEtaMin; lEta <= lEtaMax; lEta += mEtaIncrement )
	{
		for ( int lPhi = lPhiMin; lPhi <= lPhiMax; lPhi += mPhiIncrement )
		{
			this->algorithm( lEta, lPhi );
		}
	}


#ifdef __GNUG__
	if( mVerbosity )
	{
		int lStatus=0;
		std::cout << "Algorithm "
				<< abi::__cxa_demangle(typeid(*this).name(), 0,0, &lStatus)
				<< " converted " 
				<< mInputCollection->size()
				<< " x "
				<< abi::__cxa_demangle(typeid(typename tInputCollection::value_type).name(), 0,0, &lStatus)
				<< " into "
				<< mOutputCollection->size() 
				<< " x " 
				<< abi::__cxa_demangle(typeid(typename tOutputCollection::value_type).name(), 0,0, &lStatus)
				<< std::endl;
	}
#endif

	iEvent.put( mOutputCollection );

}

#endif
