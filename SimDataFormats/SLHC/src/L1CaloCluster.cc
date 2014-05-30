#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1TowerNav.h"
#include <stdlib.h>

namespace l1slhc
{

	L1CaloCluster::L1CaloCluster(  ):mIeta( 0 ),
	mIphi( 0 ),
	mEmEt( 0 ),
	mHadEt( 0 ),
	mLeadTowerE( 0 ),			       
	mSecondTowerE( 0 ),
	mThirdTowerE( 0 ),
	mFourthTowerE( 0 ),
	mRing1E( 0 ),
	mRing2E( 0 ),
	mRing3E( 0 ),
	mRing4E( 0 ),
	mFg( false ),
	mEgamma( false ),
	mCentral( false ),
	mIsoeg( false ),
	mLeadtowertau( false ), 
	mIsotau( false ), 
	mEgammavalue( 0 ), 
	mInnereta( 0 ), 
	mInnerphi( 0 ), 
	mIsoclusterseg( 0 ), 
	mIsoclusterstau( 0 ), 
	mP4( math::PtEtaPhiMLorentzVector( 0.001, 0, 0, 0. ) ),
	mIsoEmEtEG( 0 ),
	mIsoHadEtEG( 0 )
	{
	}

	L1CaloCluster::L1CaloCluster( const int &iEta, const int &iPhi ):mIeta( iEta ),
	mIphi( iPhi ),
	mEmEt( 0 ),
	mHadEt( 0 ),
        mLeadTowerE( 0 ),			       
        mSecondTowerE( 0 ),			       
        mThirdTowerE( 0 ),			       
        mFourthTowerE( 0 ),
	mRing1E( 0 ),
	mRing2E( 0 ),
	mRing3E( 0 ),
	mRing4E( 0 ),			       
	mFg( false ),
	mEgamma( false ),
	mCentral( false ),
	mIsoeg( false ),
	mLeadtowertau( false ), 
	mIsotau( false ), 
	mEgammavalue( 0 ), 
	mInnereta( 0 ), 
	mInnerphi( 0 ), 
	mIsoclusterseg( 0 ), 
	mIsoclusterstau( 0 ), 
	mP4( math::PtEtaPhiMLorentzVector( 0.001, 0, 0, 0. ) ),
	mIsoEmEtEG( 0 ),
	mIsoHadEtEG( 0 )
	{
	}



	L1CaloCluster::~L1CaloCluster(  )
	{
	}


	const int &L1CaloCluster::iEta(  ) const
	{
		return mIeta;
	}

	const int &L1CaloCluster::iPhi(  ) const
	{
		return mIphi;
	}

	const int &L1CaloCluster::innerEta(  ) const
	{
		return mInnereta;
	}

	const int &L1CaloCluster::innerPhi(  ) const
	{
		return mInnerphi;
	}


	const int L1CaloCluster::Et( int mode ) const
	{
               int returnEt=0;
	       if(mode&0x1) returnEt+=mEmEt;
               if(mode&0x2) returnEt+=mHadEt;
               return returnEt; 
	}

        const int &L1CaloCluster::EmEt(  ) const
	{
               return mEmEt; 
	}

        const int &L1CaloCluster::HadEt(  ) const
	{
               return mHadEt; 
	}

	const int &L1CaloCluster::LeadTowerE(  ) const
	{
		return mLeadTowerE;
	}

	const int &L1CaloCluster::SecondTowerE(  ) const
	{
		return mSecondTowerE;
	}

	const int &L1CaloCluster::ThirdTowerE(  ) const
	{
		return mThirdTowerE;
	}

	const int &L1CaloCluster::FourthTowerE(  ) const
	{
		return mFourthTowerE;
	}


	const int &L1CaloCluster::Ring1E(  ) const
	{
		return mRing1E;
	}


	const int &L1CaloCluster::Ring2E(  ) const
	{
		return mRing2E;
	}

	const int &L1CaloCluster::Ring3E(  ) const
	{
		return mRing3E;
	}

	const int &L1CaloCluster::Ring4E(  ) const
	{
		return mRing4E;
	}

	const bool & L1CaloCluster::fg(  ) const
	{
		return mFg;
	}

	const bool & L1CaloCluster::eGamma(  ) const
	{
		return mEgamma;
	}

	const bool & L1CaloCluster::hasLeadTower(  ) const
	{
		return mLeadtowertau;
	}


	const int &L1CaloCluster::eGammaValue(  ) const
	{
		return mEgammavalue;
	}

	const bool & L1CaloCluster::isoEG(  ) const
	{
		return mIsoeg;
	}

	const bool & L1CaloCluster::isoTau(  ) const
	{
		return mIsotau;
	}

	const bool & L1CaloCluster::isCentral(  ) const
	{
		return mCentral;
	}

	const int &L1CaloCluster::isoClustersEG(  ) const
	{
		return mIsoclusterseg;
	}

	const int &L1CaloCluster::isoClustersTau(  ) const
	{
		return mIsoclusterstau;
	}

	const int &L1CaloCluster::isoEnergyEG(  ) const
	{
		return mIsoenergyeg;
	}

	const int &L1CaloCluster::isoEnergyTau(  ) const
	{
		return mIsoenergytau;
	}
 
        const int &L1CaloCluster::isoEmEtEG(  ) const 
        {
                return mIsoEmEtEG;
        }

        const int &L1CaloCluster::isoHadEtEG(  ) const 
        {
                return mIsoHadEtEG;
        }

	bool L1CaloCluster::isEGamma(  ) const
	{
		return ( !fg(  ) && eGamma(  ) && isCentral(  ) );
	}

	bool L1CaloCluster::isIsoEGamma(  ) const
	{
		return ( !fg(  ) && eGamma(  ) && isoEG(  ) && isCentral(  ) );
	}

	bool L1CaloCluster::isIsoTau(  ) const
	{
		return hasLeadTower(  ) && isoTau(  ) && isCentral(  );
	}

	bool L1CaloCluster::isTau(  ) const
	{
	       return hasLeadTower(  ) && isCentral(  );
	}


	void L1CaloCluster::setEmEt( const int &E )
	{
		mEmEt = E;
	}
 
 	void L1CaloCluster::setHadEt( const int &E )
	{
		mHadEt = E;
	}

	void L1CaloCluster::setLeadTowerE( const int &E )
	{
		mLeadTowerE = E;
	}

	void L1CaloCluster::setSecondTowerE( const int &E )
	{
		mSecondTowerE = E;
	}

	void L1CaloCluster::setThirdTowerE( const int &E )
	{
		mThirdTowerE = E;
	}

	void L1CaloCluster::setFourthTowerE( const int &E )
	{
		mFourthTowerE = E;
	}


	void L1CaloCluster::setRing1E( const int &Ring1E )
	{
		mRing1E = Ring1E;
	}


	void L1CaloCluster::setRing2E( const int &Ring2E )
	{
		mRing2E = Ring2E;
	}

	void L1CaloCluster::setRing3E( const int &Ring3E )
	{
		mRing3E = Ring3E;
	}

	void L1CaloCluster::setRing4E( const int &Ring4E )
	{
		mRing4E = Ring4E;
	}

	void L1CaloCluster::setConstituents( const L1CaloTowerRefVector & cons )
	{
		mConstituents = cons;
	}

	const L1CaloTowerRefVector & L1CaloCluster::getConstituents(  ) const
	{
		return mConstituents;
	}


	void L1CaloCluster::setFg( const bool & fg )
	{
		mFg = fg;
	}


	void L1CaloCluster::setEGamma( const bool & eg )
	{
		mEgamma = eg;
	}

	void L1CaloCluster::setLeadTower( const bool & eg )
	{
		mLeadtowertau = eg;
	}


	void L1CaloCluster::setEGammaValue( const int &eg )
	{
		mEgammavalue = eg;
	}

	void L1CaloCluster::setIsoEG( const bool & eg )
	{
		mIsoeg = eg;
	}

	void L1CaloCluster::setIsoTau( const bool & eg )
	{
		mIsotau = eg;
	}

	void L1CaloCluster::setIsoClusters( const int &eg, const int &tau )
	{
		mIsoclusterseg = eg;
		mIsoclusterstau = tau;
	}

	void L1CaloCluster::setIsoEnergy( const int &eg, const int &tau )
	{
		mIsoenergyeg = eg;
		mIsoenergytau = tau;
	}

        void L1CaloCluster::setIsoEmAndHadEtEG(const int& isoEmEt,const int& isoHadEt)
        {
	        mIsoEmEtEG = isoEmEt;
                mIsoHadEtEG = isoHadEt;  
        }
	void L1CaloCluster::setCentral( const bool & eg )
	{
		mCentral = eg;
	}

	void L1CaloCluster::setPosBits( const int &eta, const int &phi )
	{
		mInnereta = eta;
		mInnerphi = phi;

	}

	void L1CaloCluster::setLorentzVector( const math::PtEtaPhiMLorentzVector & v )
	{
		mP4 = v;
	}


	void L1CaloCluster::addConstituent( const L1CaloTowerRef & tower )
	{
		mEmEt += tower->E(  );
		mHadEt += tower->H(  ); 
		mConstituents.push_back( tower );
	}

	int L1CaloCluster::hasConstituent( const int &eta, const int &phi ) const
	{
		for ( unsigned int i = 0; i < mConstituents.size(  ); ++i )
		{
			L1CaloTowerRef tower = mConstituents.at( i );
			if ( tower->iEta(  ) == L1TowerNav::getOffsetIEta(mIeta,eta) && tower->iPhi(  ) == L1TowerNav::getOffsetIPhi(L1TowerNav::getOffsetIEta(mIeta,eta),mIphi,phi) ) //SHarper change: fix iEta -ve to +ve and iPhi 72->1 bug, warning need to check behavour when the offset eta is on a phi change boundary
			{
				return i;
			}
		}
		return -1;
	}


	L1CaloTowerRef L1CaloCluster::getConstituent( const int &pos ) const
	{
		return mConstituents.at( pos );
	}



	void L1CaloCluster::removeConstituent( const int &eta, const int &phi )
	{

		int pos = hasConstituent( eta, phi );

		if ( pos != -1 )
		{
			mEmEt-=mConstituents.at( pos )->E(  );
			mHadEt-=mConstituents.at( pos )->H(  ); 
			mConstituents.erase( mConstituents.begin(  ) + pos );
		}
	}



	const math::PtEtaPhiMLorentzVector & L1CaloCluster::p4(  ) const
	{
		return mP4;
	}

}


namespace std
{
	bool operator<( const l1slhc::L1CaloCluster & aLeft, const l1slhc::L1CaloCluster & aRight )
	{
	  if ( aLeft.EmEt() + aLeft.HadEt() == aRight.EmEt(  ) + aRight.HadEt() )
		{
			// for two objects with equal energy, favour the more central one
			return ( abs( aLeft.iEta(  ) ) > abs( aRight.iEta(  ) ) );
		}
		else
		{
		  return ( aLeft.EmEt()+aLeft.HadEt() < aRight.EmEt() + aRight.HadEt() );
		}
	}
}


// pretty print
std::ostream & operator<<( std::ostream & aStream, const l1slhc::L1CaloCluster & aL1CaloCluster )
{
	aStream << "L1CaloCluster"
		<< " iEta=" << aL1CaloCluster.iEta(  )
		<< " iPhi=" << aL1CaloCluster.iPhi(  )
		<< " E=" << aL1CaloCluster.EmEt(  ) + aL1CaloCluster.HadEt( )
		<< " eta=" << aL1CaloCluster.p4(  ).eta(  )
		<< " phi=" << aL1CaloCluster.p4(  ).phi(  )
		<< " pt=" << aL1CaloCluster.p4(  ).pt(  )
		<< " egamma=" << aL1CaloCluster.eGammaValue(  )
		<< " central=" << aL1CaloCluster.isCentral(  ) 
		<< " fg=" << aL1CaloCluster.fg(  ) 
		<< "\n with constituents:\n";
	for ( l1slhc::L1CaloTowerRefVector::const_iterator i = aL1CaloCluster.getConstituents(  ).begin(  ); i != aL1CaloCluster.getConstituents(  ).end(  ); ++i )
		aStream << "  iEta=" << ( **i ).iEta(  ) 
			<< " iPhi=" << ( **i ).iPhi(  ) 
			<< " ET=" << ( **i ).E(  )
			<< "\n";
	return aStream;
}
