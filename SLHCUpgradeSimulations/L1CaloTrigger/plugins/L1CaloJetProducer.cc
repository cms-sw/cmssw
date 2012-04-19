
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloJetFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegion.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegionFwd.h"

class L1CaloJetProducer:
public L1CaloAlgoBase < l1slhc::L1CaloRegionCollection, l1slhc::L1CaloJetCollection >
{
  public:
	L1CaloJetProducer( const edm::ParameterSet & );
	 ~L1CaloJetProducer(  );

	// void initialize( );

	void algorithm( const int &, const int & );

  private:
	void calculateJetPosition( l1slhc::L1CaloJet & lJet );

};

L1CaloJetProducer::L1CaloJetProducer( const edm::ParameterSet & aConfig ):L1CaloAlgoBase < l1slhc::L1CaloRegionCollection, l1slhc::L1CaloJetCollection > ( aConfig )
{
	mPhiOffset = -7;
	mEtaOffset = -7;
	mPhiIncrement = 4;
	mEtaIncrement = 4;
}

L1CaloJetProducer::~L1CaloJetProducer(  )
{
}

/* 
   void L1CaloJetProducer::initialize( ) { }
*/

void L1CaloJetProducer::algorithm( const int &aEta, const int &aPhi )
{


	int lRegionIndex = mCaloTriggerSetup->getBin( aEta, aPhi );
	std::pair < int, int > lRegionEtaPhi = mCaloTriggerSetup->getTowerEtaPhi( lRegionIndex );


	l1slhc::L1CaloJet lJet( lRegionEtaPhi.first, lRegionEtaPhi.second );

/*
	for ( int lRegionEta = aEta; lRegionEta <= aEta + 4; lRegionEta+=4 )
	{
		for ( int lRegionPhi = aPhi; lRegionPhi <= aPhi + 4; lRegionPhi+=4 )
		{

			l1slhc::L1CaloRegionCollection::const_iterator lRegionItr = fetch( lRegionEta, lRegionPhi );
*/

	for ( int lRegionEta = 0; lRegionEta != 2; ++lRegionEta )
	{
		for ( int lRegionPhi = 0; lRegionPhi != 2; ++lRegionPhi )
		{
			l1slhc::L1CaloRegionCollection::const_iterator lRegionItr = fetch( aEta+(lRegionEta<<2) , aPhi+(lRegionPhi<<2) );
			if ( lRegionItr != mInputCollection->end(  ) )
			{
				l1slhc::L1CaloRegionRef lRef( mInputCollection, lRegionItr - mInputCollection->begin(  ) );
				lJet.addConstituent( lRef );
			}
		}
	}

	if ( lJet.E(  ) > 0 )
	{
		calculateJetPosition( lJet );
		mOutputCollection->insert( lRegionEtaPhi.first, lRegionEtaPhi.second, lJet );
	}

}






void L1CaloJetProducer::calculateJetPosition( l1slhc::L1CaloJet & lJet )
{

	// Calculate float value of eta for barrel+endcap(L.Gray)
	double eta = -1982.;		// an important year...
	double etaOffset = 0.0435;  //0.087 / 2.0;
	int abs_eta = abs( lJet.iEta(  ) + 4 );

	if ( abs_eta <= 20 )
	{
		eta = ( abs_eta * 0.0870 ) - etaOffset;
	}
	else
	{
		const double endcapEta[8] = { 0.09, 0.1, 0.113, 0.129, 0.15, 0.178, 0.15, 0.35 };
		//int offset = abs( lJet.iEta(  ) + 4 ) - 21;
		abs_eta -= 21;
	
		eta = 1.74; //( 20 * 0.0870 );	// -etaOffset;
//		for ( int i = 0; i <= offset; ++i )
		for ( int i = 0; i <= abs_eta; ++i )
		{
			eta += endcapEta[i];
		}
//		eta -= endcapEta[abs( lJet.iEta(  ) + 4 ) - 21] / 2.;
		eta -= endcapEta[abs_eta] / 2.;
	}

//	if ( lJet.iEta(  ) + 4 < 0 )
	if ( lJet.iEta(  ) < -4 )
		eta = -eta;

//	double phi = ( ( lJet.iPhi(  ) + 4 ) * 0.087 ) - 0.087 / 2.;
	double phi = ( lJet.iPhi(  ) * 0.087 ) + 0.3045;
	double Et = double( lJet.E(  ) ) / 2.;

	lJet.setP4( math::PtEtaPhiMLorentzVector( Et, eta, phi, 0. ) );
}



DEFINE_EDM_PLUGIN( edm::MakerPluginFactory, edm::WorkerMaker < L1CaloJetProducer >, "L1CaloJetProducer" );
DEFINE_FWK_PSET_DESC_FILLER( L1CaloJetProducer );
