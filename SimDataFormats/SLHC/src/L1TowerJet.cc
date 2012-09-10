#include "SimDataFormats/SLHC/interface/L1TowerJet.h"
#include <stdlib.h>

namespace l1slhc
{

	L1TowerJet::L1TowerJet( ):
	mIeta( 0 ), 
	mIphi( 0 ), 
	mE( 0 ), 
	mCentral( true ),
	mJetSize( 12 ),
	mJetShapeType( square )
	{
	}

	L1TowerJet::L1TowerJet( const int& aJetSize, const L1TowerJet::tJetShape& aJetShapeType ):
	mIeta( 0 ), 
	mIphi( 0 ), 
	mE( 0 ), 
	mCentral( true ),
	mJetSize( aJetSize ),
	mJetShapeType( aJetShapeType )
	{
	}

	L1TowerJet::L1TowerJet( const int& aJetSize, const L1TowerJet::tJetShape& aJetShapeType , const int &iEta, const int &iPhi ):
	mIeta( iEta ), 
	mIphi( iPhi ), 
	mE( 0 ), 
	mCentral( true ),
	mJetSize( aJetSize ),
	mJetShapeType( aJetShapeType )
	{
	}

	L1TowerJet::~L1TowerJet(  )
	{
	}




	void L1TowerJet::setP4( const math::PtEtaPhiMLorentzVector & p4 )
	{
		mP4 = p4;
	}

	void L1TowerJet::setCentral( const bool & central )
	{
		mCentral = central;
	}
/*
	void L1TowerJet::setE( const int &E )
	{
		mE = E;
	}
*/


	const int &L1TowerJet::iEta(  ) const
	{
		return mIeta;
	}

	const int &L1TowerJet::iPhi(  ) const
	{
		return mIphi;
	}

	const int &L1TowerJet::E(  ) const
	{
		return mE;
	}

	const bool & L1TowerJet::central(  ) const
	{
		return mCentral;
	}

	const math::PtEtaPhiMLorentzVector & L1TowerJet::p4(  ) const
	{
		return mP4;
	}

	const int& L1TowerJet::JetSize(  ) const
	{
		return mJetSize;
	}

	const L1TowerJet::tJetShape& L1TowerJet::JetShape(  ) const
	{
		return mJetShapeType;
	}



	double L1TowerJet::EcalVariance(  ) const
	{
		double lMean(0.0);
		double lMeanSq(0.0);

		for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
			lMean += (**lConstituentIt).E();
			lMeanSq += ((**lConstituentIt).E() * (**lConstituentIt).E());
		}

		lMean /= mConstituents.size();	
		lMeanSq /= mConstituents.size();	

		return lMeanSq - (lMean*lMean);
	}


	double L1TowerJet::HcalVariance(  ) const
	{
		double lMean(0.0);
		double lMeanSq(0.0);

		for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
			lMean += (**lConstituentIt).H();
			lMeanSq += ((**lConstituentIt).H() * (**lConstituentIt).H());
		}

		lMean /= mConstituents.size();	
		lMeanSq /= mConstituents.size();	

		return lMeanSq - (lMean*lMean);
	}


	double L1TowerJet::EnergyVariance(  ) const
	{
		double lMean( double(mE) / double(mConstituents.size()) );
		double lMeanSq(0.0);

		double lTower;
		for ( L1CaloTowerRefVector::const_iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt ){
			lTower = (**lConstituentIt).E() + (**lConstituentIt).H();
			lMeanSq += ( lTower * lTower );
		}

		lMeanSq /= mConstituents.size();	

		return lMeanSq - (lMean*lMean);
	}









	void L1TowerJet::addConstituent( const L1CaloTowerRef & Tower )
	{
		mE += Tower->E(  );
		mE += Tower->H(  );
		mConstituents.push_back( Tower );
	}

	void L1TowerJet::removeConstituent( const int &eta, const int &phi )
	{
		L1CaloTowerRefVector::iterator lConstituent = getConstituent( eta, phi );
		if ( lConstituent != mConstituents.end() )
		{
			mE -= (**lConstituent).E(  );
			mE -= (**lConstituent).H(  );
			mConstituents.erase( lConstituent );
		}
	}

	const L1CaloTowerRefVector & L1TowerJet::getConstituents(  ) const
	{
		return mConstituents;
	}


	L1CaloTowerRefVector::iterator L1TowerJet::getConstituent( const int &eta, const int &phi )
	{
		for ( L1CaloTowerRefVector::iterator lConstituentIt = mConstituents.begin() ; lConstituentIt != mConstituents.end(); ++lConstituentIt )
			if ( (**lConstituentIt).iEta(  ) == eta + mIeta && (**lConstituentIt).iPhi(  ) == phi + mIphi )
				return lConstituentIt;

		return mConstituents.end();
	}









}




namespace std
{
	bool operator<( const l1slhc::L1TowerJet & aLeft, const l1slhc::L1TowerJet & aRight )
	{
		if ( aLeft.E(  ) == aRight.E(  ) )
		{
			// for two objects with equal energy, favour the more central one
			return ( abs( aLeft.iEta(  ) ) > abs( aRight.iEta(  ) ) );
		}
		else
		{
			return ( aLeft.E(  ) < aRight.E(  ) );
		}
	}
}


// pretty print
std::ostream & operator<<( std::ostream & aStream, const l1slhc::L1TowerJet & aL1TowerJet )
{
	aStream << "L1TowerJet" 
		<< " iEta=" << aL1TowerJet.iEta(  ) 
		<< " iPhi=" << aL1TowerJet.iPhi(  ) 
		<< "\n with constituents:\n";
	for ( l1slhc::L1CaloTowerRefVector::const_iterator i = aL1TowerJet.getConstituents(  ).begin(  ); i < aL1TowerJet.getConstituents(  ).end(  ); ++i )
		aStream << "  iEta=" << ( **i ).iEta(  ) 
			<< " iPhi=" << ( **i ).iPhi(  ) 
			<< " ET=" << ( **i ).E(  ) 
			<< "\n";
	return aStream;
}
