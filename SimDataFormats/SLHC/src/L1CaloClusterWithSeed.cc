#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeed.h"
#include "SimDataFormats/SLHC/interface/L1TowerNav.h"

#include <stdlib.h>

#include <iostream>

namespace l1slhc
{

    L1CaloClusterWithSeed::L1CaloClusterWithSeed(  ):
        mSeedThreshold(4),
        mEmThreshold(2),
        mHadThreshold(2),
        mIeta( 0 ),
        mIphi( 0 ),
        mEmEt( 0 ),
        mHadEt( 0 ),
        mTrimmedPlus(false),
        mTrimmedMinus(false),
        mFg( false ),
        mEgamma( false ),
        mEgammavalue( 0 ), 
        mIsoeg( false ),
        mIsoEmEtEG( 0 ),
        mIsoHadEtEG( 0 ),
        mInnereta( 0 ), 
        mInnerphi( 0 ), 
        mP4( math::PtEtaPhiMLorentzVector( 0.001, 0, 0, 0. ) )
    {
    }

    L1CaloClusterWithSeed::L1CaloClusterWithSeed( const int &iEta, const int &iPhi ):
        mSeedThreshold(2),
        mEmThreshold(2),
        mHadThreshold(2),
        mIeta( iEta ),
        mIphi( iPhi ),
        mEmEt( 0 ),
        mHadEt( 0 ),
        mTrimmedPlus(false),
        mTrimmedMinus(false),
        mFg( false ),
        mEgamma( false ),
        mEgammavalue( 0 ), 
        mIsoeg( false ),
        mIsoEmEtEG( 0 ),
        mIsoHadEtEG( 0 ),
        mInnereta( 0 ), 
        mInnerphi( 0 ), 
        mP4( math::PtEtaPhiMLorentzVector( 0.001, 0, 0, 0. ) )
    {
    }

    L1CaloClusterWithSeed::L1CaloClusterWithSeed( const L1CaloTowerRef & seed, int seedThreshold, int emThreshold, int hadThreshold ):
        mSeedTower(seed),
        mSeedThreshold(seedThreshold),
        mEmThreshold(emThreshold),
        mHadThreshold(hadThreshold),
        mIeta( seed->iEta() ),
        mIphi( seed->iPhi() ),
        mEmEt( seed->E() ),
        mHadEt( (seed->H()>=hadThreshold ? seed->H() : 0.) ),
        mTrimmedPlus(false),
        mTrimmedMinus(false),
        mFg( seed->EcalFG() ),
        mEgamma( false ),
        mEgammavalue( 0 ), 
        mIsoeg( false ),
        mIsoEmEtEG( 0 ),
        mIsoHadEtEG( 0 ),
        mInnereta( 0 ), 
        mInnerphi( 0 ), 
        mP4( math::PtEtaPhiMLorentzVector( 0.001, 0, 0, 0. ) )
    {
        if(seed->E()<mSeedThreshold)
        {
            std::cout<<"L1CaloClusterWithSeed: WARNING: Trying to seed a cluster with a tower below the seeding threshold\n";
        }
    }



    L1CaloClusterWithSeed::~L1CaloClusterWithSeed(  )
    {
    }


    int L1CaloClusterWithSeed::iEta(  ) const
    {
        return mIeta;
    }

    int L1CaloClusterWithSeed::iPhi(  ) const
    {
        return mIphi;
    }

    int L1CaloClusterWithSeed::innerEta(  ) const
    {
        return mInnereta;
    }

    int L1CaloClusterWithSeed::innerPhi(  ) const
    {
        return mInnerphi;
    }


    int L1CaloClusterWithSeed::Et(int mode) const
    {
        int returnEt=0;
        if(mode&0x1) returnEt += mEmEt;
        if(mode&0x2) returnEt += mHadEt;
        return returnEt;
    }

    int L1CaloClusterWithSeed::EmEt(  ) const
    {
        return mEmEt;
    }

    int L1CaloClusterWithSeed::HadEt(  ) const
    {
        return mHadEt;
    }

    bool L1CaloClusterWithSeed::trimmedPlus(  ) const
    {
        return mTrimmedPlus;
    }

    bool L1CaloClusterWithSeed::trimmedMinus(  ) const
    {
        return mTrimmedMinus;
    }


    bool L1CaloClusterWithSeed::fg(  ) const
    {
        return mFg;
    }

    bool L1CaloClusterWithSeed::eGamma(  ) const
    {
        return mEgamma;
    }

    int L1CaloClusterWithSeed::eGammaValue(  ) const
    {
        return mEgammavalue;
    }


    bool L1CaloClusterWithSeed::isEGamma(  ) const
    {
        return ( !(seedEmEt()>6 && fg(  )) && eGamma(  ) );
    }

    const bool & L1CaloClusterWithSeed::isoEG(  ) const
    {
        return mIsoeg;
    }

    const int &L1CaloClusterWithSeed::isoEmEtEG(  ) const 
    {
        return mIsoEmEtEG;
    }

    const int &L1CaloClusterWithSeed::isoHadEtEG(  ) const 
    {
        return mIsoHadEtEG;
    }

    bool L1CaloClusterWithSeed::isIsoEGamma(  ) const
    {
        return ( !(seedEmEt()>6 && fg(  )) && eGamma(  ) && isoEG(  ) );
    }

    void L1CaloClusterWithSeed::setEmEt( int E )
    {
        mEmEt = E;
    }

    void L1CaloClusterWithSeed::setHadEt( int H )
    {
        mHadEt = H;
    }

    void L1CaloClusterWithSeed::setTrimmedPlus( bool trimmed)
    {
        mTrimmedPlus = trimmed;
    }

    void L1CaloClusterWithSeed::setTrimmedMinus( bool trimmed)
    {
        mTrimmedMinus = trimmed;
    }
    

    void L1CaloClusterWithSeed::setConstituents( const L1CaloTowerRefVector & cons )
    {
        mConstituents = cons;
    }

    const L1CaloTowerRefVector & L1CaloClusterWithSeed::getConstituents(  ) const
    {
        return mConstituents;
    }


    void L1CaloClusterWithSeed::setFg( bool fg )
    {
        mFg = fg;
    }


    void L1CaloClusterWithSeed::setEGamma( bool eg )
    {
        mEgamma = eg;
    }


    void L1CaloClusterWithSeed::setEGammaValue( int eg )
    {
        mEgammavalue = eg;
    }

    void L1CaloClusterWithSeed::setIsoEG( const bool & eg )
    {
        mIsoeg = eg;
    }

    void L1CaloClusterWithSeed::setIsoEmAndHadEtEG(const int& isoEmEt,const int& isoHadEt)
    {
        mIsoEmEtEG = isoEmEt;
        mIsoHadEtEG = isoHadEt;  
    }

    void L1CaloClusterWithSeed::setPosBits( int eta, int phi )
    {
        mInnereta = eta;
        mInnerphi = phi;

    }

    void L1CaloClusterWithSeed::setLorentzVector( const math::PtEtaPhiMLorentzVector & v )
    {
        mP4 = v;
    }


    void L1CaloClusterWithSeed::addConstituent( const L1CaloTowerRef & tower )
    {
        mEmEt  += (tower->E()>=mEmThreshold ? tower->E(  ) : 0);
        mHadEt += (tower->H()>=mHadThreshold ? tower->H(  ) : 0);
        mConstituents.push_back( tower );
        mConstituentSharing.push_back(0);
    }

    // Friend towers are not included in the cluster. But may be included in later steps of clustering.
    void L1CaloClusterWithSeed::addFriend( const L1CaloTowerRef & tower )
    {
        mFriends.push_back( tower );
    }

    int L1CaloClusterWithSeed::hasConstituent( int eta, int phi ) const
    {
        for ( unsigned int i = 0; i < mConstituents.size(  ); ++i )
        {
            L1CaloTowerRef tower = mConstituents.at( i );
            //std::cout<<"ieta="<<tower->iEta()<<"("<<mIeta+eta<<")\n";
            //if ( tower->iEta(  ) == mIeta + eta && tower->iPhi(  ) == mIphi + phi )
            if ( tower->iEta(  ) == L1TowerNav::getOffsetIEta(mIeta,eta) && tower->iPhi(  ) == L1TowerNav::getOffsetIPhi(L1TowerNav::getOffsetIEta(mIeta,eta),mIphi,phi) ) //SHarper change: fix iEta -ve to +ve and iPhi 72->1 bug, warning need to check behavour when the offset eta is on a phi change boundary
            {
                return i;
            }
        }
        return -1;
    }

    int L1CaloClusterWithSeed::hasFriend( int eta, int phi )
    {
        for ( unsigned int i = 0; i < mFriends.size(  ); ++i )
        {
            L1CaloTowerRef tower = mFriends.at( i );
            //if ( tower->iEta(  ) == mIeta + eta && tower->iPhi(  ) == mIphi + phi )
            if ( tower->iEta(  ) == L1TowerNav::getOffsetIEta(mIeta,eta) && tower->iPhi(  ) == L1TowerNav::getOffsetIPhi(L1TowerNav::getOffsetIEta(mIeta,eta),mIphi,phi) ) //SHarper change: fix iEta -ve to +ve and iPhi 72->1 bug, warning need to check behavour when the offset eta is on a phi change boundary
            {
                return i;
            }
        }
        return -1;
    }

    L1CaloTowerRef L1CaloClusterWithSeed::getSeedTower() const
    {
        return mSeedTower;
    }

    L1CaloTowerRef L1CaloClusterWithSeed::getConstituent( int pos )
    {
        return mConstituents.at( pos );
    }

    L1CaloTowerRef L1CaloClusterWithSeed::getFriend( int pos )
    {
        return mFriends.at( pos );
    }



    void L1CaloClusterWithSeed::removeConstituent( int eta, int phi )
    {

        int pos = hasConstituent( eta, phi );

        if ( pos != -1 )
        {
            mEmEt  -= constituentEmEt( eta, phi );
            mHadEt -= constituentHadEt( eta, phi );
            mConstituents.erase( mConstituents.begin(  ) + pos );
            mConstituentSharing.erase( mConstituentSharing.begin(  ) + pos );
        }
    }

    void L1CaloClusterWithSeed::removeFriend( int eta, int phi )
    {

        int pos = hasFriend( eta, phi );

        if ( pos != -1 )
        {
            mFriends.erase( mFriends.begin(  ) + pos );
        }
    }


    int L1CaloClusterWithSeed::seedEmEt() const
    {
        return (mSeedTower->E()>=mSeedThreshold ? mSeedTower->E() : 0);
    }

    int L1CaloClusterWithSeed::seedHadEt() const
    {
        return (mSeedTower->H()>=mHadThreshold ? mSeedTower->H() : 0.);
    }

    int L1CaloClusterWithSeed::constituentEmEt(int eta, int phi) const
    {
        int pos = hasConstituent( eta, phi );

        int lConstituentE = 0;
        if ( pos != -1 )
        {
            lConstituentE = (mConstituents[pos]->E()>=mEmThreshold ? mConstituents[pos]->E() : 0);
            int lSharing = mConstituentSharing[pos];
            switch(lSharing)
            {
                case 1:
                    lConstituentE = lConstituentE - lConstituentE/4;
                    break;
                case 2: 
                    lConstituentE = lConstituentE - lConstituentE/2;
                    break;
                case 3:
                    lConstituentE = lConstituentE/2;
                    break;
                case 4:
                    lConstituentE = lConstituentE/4;
                    break;
                case 5:
                    lConstituentE = 0;
                    break;
                default:
                    break;
            };
        }
        return lConstituentE;
    }

    int L1CaloClusterWithSeed::constituentHadEt(int eta, int phi) const
    {
        int pos = hasConstituent( eta, phi );

        int lConstituentH = 0;
        if ( pos != -1 )
        {
            lConstituentH = (mConstituents[pos]->H()>=mHadThreshold ? mConstituents[pos]->H() : 0);
            int lSharing = mConstituentSharing[pos];
            switch(lSharing)
            {
                case 1:
                    lConstituentH = lConstituentH - lConstituentH/4;
                    break;
                case 2: 
                    lConstituentH = lConstituentH - lConstituentH/2;
                    break;
                case 3:
                    lConstituentH = lConstituentH/2;
                    break;
                case 4:
                    lConstituentH = lConstituentH/4;
                    break;
                case 5:
                    lConstituentH = 0;
                    break;
                default:
                    break;
            };
        }
        return lConstituentH;
    }

    void L1CaloClusterWithSeed::shareConstituent(int eta, int phi, int sharing)
    {
        int pos = hasConstituent( eta, phi );
        if ( pos != -1 )
        {
            int cEold = constituentEmEt(eta,phi);
            int cHold = constituentHadEt(eta,phi);
            mConstituentSharing[pos] = sharing;
            int cEnew = constituentEmEt(eta, phi);
            int cHnew = constituentHadEt(eta, phi);
            mEmEt  = mEmEt - cEold + cEnew;
            mHadEt = mHadEt - cHold + cHnew;
        }
    }


    const math::PtEtaPhiMLorentzVector & L1CaloClusterWithSeed::p4(  ) const
    {
        return mP4;
    }

}


namespace std
{
    bool operator<( const l1slhc::L1CaloClusterWithSeed & aLeft, const l1slhc::L1CaloClusterWithSeed & aRight )
    {
        if ( aLeft.EmEt()+aLeft.HadEt() == aRight.EmEt()+aRight.HadEt() )
        {
            // for two objects with equal energy, favour the more central one
            return ( abs( aLeft.iEta() ) > abs( aRight.iEta() ) );
        }
        else
        {
            return ( aLeft.EmEt()+aLeft.HadEt() < aRight.EmEt()+aRight.HadEt() );
        }
    }
}


// pretty print
std::ostream & operator<<( std::ostream & aStream, const l1slhc::L1CaloClusterWithSeed & aL1CaloClusterWithSeed )
{
    aStream << "L1CaloClusterWithSeed"
        << " iEta=" << aL1CaloClusterWithSeed.iEta(  )
        << " iPhi=" << aL1CaloClusterWithSeed.iPhi(  )
        << " E=" << aL1CaloClusterWithSeed.EmEt(  )
        << " H=" << aL1CaloClusterWithSeed.HadEt(  )
        << " eta=" << aL1CaloClusterWithSeed.p4(  ).eta(  )
        << " phi=" << aL1CaloClusterWithSeed.p4(  ).phi(  )
        << " pt=" << aL1CaloClusterWithSeed.p4(  ).pt(  )
        << " egamma=" << aL1CaloClusterWithSeed.eGammaValue(  )
        << " fg=" << aL1CaloClusterWithSeed.fg(  ) 
        << "\n with constituents:\n";
    for ( l1slhc::L1CaloTowerRefVector::const_iterator i = aL1CaloClusterWithSeed.getConstituents(  ).begin(  ); i != aL1CaloClusterWithSeed.getConstituents(  ).end(  ); ++i )
        aStream << "  iEta=" << ( **i ).iEta(  ) 
            << " iPhi=" << ( **i ).iPhi(  ) 
            << " ET=" << ( **i ).E(  )
            << "\n";
    return aStream;
}
