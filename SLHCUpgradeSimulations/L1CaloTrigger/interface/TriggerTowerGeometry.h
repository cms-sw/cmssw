#ifndef TRIGGER_TOWER_GEO
#define TRIGGER_TOWER_GEO

class TriggerTowerGeometry
{
  public:
	TriggerTowerGeometry(  )
	{

	        // Calculate the TT phi width, 2*PI/PhiTTs
	        mDeltaPhi = 6.28318530718/72.;

		// Store the TT eta widths
		for ( int i = 1; i <= 20; ++i )
			mMappingeta[i] = 0.087;

		mMappingeta[21] = 0.09;
		mMappingeta[22] = 0.1;
		mMappingeta[23] = 0.113;
		mMappingeta[24] = 0.129;
		mMappingeta[25] = 0.15;
		mMappingeta[26] = 0.178;
		mMappingeta[27] = 0.15;
		mMappingeta[28] = 0.35;
	}

	double eta( const int& iEta )
	{
		double eta = 0;
		for ( int i = 1; i <= abs( iEta ); ++i )
		{
			eta += mMappingeta[i];
		}
		eta -= mMappingeta[abs( iEta )] / 2;

		if ( iEta > 0 )
			return eta;
		else
			return -eta;
	}

	double phi( const int& iPhi )
	{

	  //BUG - Should use: deltaPhi = 2*pi/72
	  //return ( double( iPhi )-0.5 ) * 0.087;
	  return ( double( iPhi ) - 0.5 ) * mDeltaPhi;
	}

	double towerEtaSize( const int& iEta )
	{
		return mMappingeta[abs( iEta )];
	}

	double towerPhiSize( const int& iPhi )
	{
	  
	  return mDeltaPhi;
	  //return 0.087;
	}

  private:
	std::map < int, double >mMappingeta;
	double mDeltaPhi;
};

#endif
