#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include <stdlib.h>

namespace l1slhc
{

	L1CaloJet::L1CaloJet(  ):
	mIeta( 0 ), 
	mIphi( 0 ), 
	mE( 0 ), 
	mCentral( true )
	{
	}



	L1CaloJet::L1CaloJet( const int &iEta, const int &iPhi ):
	mIeta( iEta ), 
	mIphi( iPhi ), 
	mE( 0 ), 
	mCentral( true )
	{
	}

	L1CaloJet::~L1CaloJet(  )
	{


	}

	void L1CaloJet::setP4( const math::PtEtaPhiMLorentzVector & p4 )
	{
		mP4 = p4;
	}

	void L1CaloJet::setCentral( const bool & central )
	{
		mCentral = central;
	}


	const int &L1CaloJet::iEta(  ) const
	{
		return mIeta;
	}


	const int &L1CaloJet::iPhi(  ) const
	{
		return mIphi;
	}


	const int &L1CaloJet::E(  ) const
	{
		return mE;
	}

	const bool & L1CaloJet::central(  ) const
	{
		return mCentral;
	}



	void L1CaloJet::setE( const int &E )
	{
		mE = E;
	}

	const math::PtEtaPhiMLorentzVector & L1CaloJet::p4(  ) const
	{
		return mP4;
	}


	void L1CaloJet::addConstituent( const L1CaloRegionRef & region )
	{
		mE += region->E(  );
		mConstituents.push_back( region );
	}

	const L1CaloRegionRefVector & L1CaloJet::getConstituents(  ) const
	{
		return mConstituents;
	}

	int L1CaloJet::hasConstituent( const int &eta, const int &phi )
	{
		int pos = -1;
		for ( unsigned int i = 0; i < mConstituents.size(  ); ++i )
		{
			L1CaloRegionRef tower = mConstituents.at( i );
			if ( tower->iEta(  ) == eta + mIeta && tower->iPhi(  ) == phi + mIphi )
			{
				pos = i;
				break;
			}
		}

		return pos;
	}

	void L1CaloJet::removeConstituent( const int &eta, const int &phi )
	{
		int pos = hasConstituent( eta, phi );
		if ( pos != -1 )
		{
			mE = mE - mConstituents.at( pos )->E(  );
			mConstituents.erase( mConstituents.begin(  ) + pos );
		}
	}
}




namespace std
{
	bool operator<( const l1slhc::L1CaloJet & aLeft, const l1slhc::L1CaloJet & aRight )
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
std::ostream & operator<<( std::ostream & aStream, const l1slhc::L1CaloJet & aL1CaloJet )
{
	aStream << "L1CaloJet" 
		<< " iEta=" << aL1CaloJet.iEta(  ) 
		<< " iPhi=" << aL1CaloJet.iPhi(  ) 
		<< "\n with constituents:\n";
	for ( l1slhc::L1CaloRegionRefVector::const_iterator i = aL1CaloJet.getConstituents(  ).begin(  ); i < aL1CaloJet.getConstituents(  ).end(  ); ++i )
		aStream << "  iEta=" << ( **i ).iEta(  ) 
			<< " iPhi=" << ( **i ).iPhi(  ) 
			<< " ET=" << ( **i ).E(  ) 
			<< "\n";
	return aStream;
}
