#include "SimDataFormats/SLHC/interface/L1CaloRegion.h"

#include <stdlib.h>


namespace l1slhc
{

	L1CaloRegion::L1CaloRegion(  ):mIeta( 0 ), 
	mIphi( 0 ), 
	mE( 0 )
	{
	}

	L1CaloRegion::L1CaloRegion( const int &iEta, const int &iPhi, const int &E ):mIeta( iEta ), 
	mIphi( iPhi ), 
	mE( E )
	{
	}


	L1CaloRegion::~L1CaloRegion(  )
	{
	}

	const int &L1CaloRegion::iEta(  ) const
	{
		return mIeta;
	}

	const int &L1CaloRegion::iPhi(  ) const
	{
		return mIphi;
	}

	const int &L1CaloRegion::E(  ) const
	{
		return mE;
	}

}


namespace std
{
	bool operator<( const l1slhc::L1CaloRegion & aLeft, const l1slhc::L1CaloRegion & aRight )
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
