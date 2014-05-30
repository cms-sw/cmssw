#ifndef L1CaloClusterWithSeed_h
#define L1CaloClusterWithSeed_h


#include <ostream>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"

namespace l1slhc
{

    class L1CaloClusterWithSeed
    {

        friend bool operator==( const l1slhc::L1CaloClusterWithSeed & a, const l1slhc::L1CaloClusterWithSeed & b )
        {
            if ( a.mIeta != b.mIeta )
                return false;
            if ( a.mIphi != b.mIphi )
                return false;
            if ( a.mFg != b.mFg )
                return false;
            if ( a.mEgamma != b.mEgamma )
                return false;
            if ( a.mEgammavalue != b.mEgammavalue )
                return false;
            if ( a.mIsoeg != b.mIsoeg )
                return false;
            if ( a.mEmEt != b.mEmEt )
                return false;
            if ( a.mHadEt != b.mHadEt )
                return false;
            if ( a.mP4 != b.mP4 )
                return false;
            return true;
        }


        public:
        L1CaloClusterWithSeed(  );
        L1CaloClusterWithSeed( const int &, const int & );
        L1CaloClusterWithSeed( const L1CaloTowerRef & seed, int seedThreshold=4, int emThreshold=2, int hadThreshold=2 );
        ~L1CaloClusterWithSeed(  );

        void setFg( bool  );	// Set FG Bit
        void setEGamma( bool  );	// Set EGamma Bit
        void setEGammaValue( int  );	// Set EGamma Value E/E+H (%)
        void setIsoEmAndHadEtEG( const int &, const int & );
		void setIsoEG( const bool & );	// EG isolation
        void setLorentzVector( const math::PtEtaPhiMLorentzVector & );	
        void setPosBits( int , int  );
        void setConstituents( const L1CaloTowerRefVector & );
        void setEmEt( int  );
        void setHadEt( int  );
        void setTrimmedPlus( bool trimmed=true);
        void setTrimmedMinus( bool trimmed=true);
        void addConstituent( const L1CaloTowerRef & );
        int hasConstituent( int , int  ) const;
        void removeConstituent( int , int  );
        void addFriend(const L1CaloTowerRef & );
        int hasFriend( int , int  );
        void removeFriend( int , int  );

        // Get Functions
        int iEta(  ) const;	// Eta of seed in integer coordinates
        int iPhi(  ) const;	// Phi of seed in integer
        int Et(int mode=0x3) const;	// Compressed Et SH: name change  mode = bit 1 add EmEt, bit 2 add HadEt, I had to make this a non-reference variable as it returns the sum of mEmEt + mHadEt, leave as is for now
        int EmEt(  ) const;	// Compressed ECAL Et 
        int HadEt(  ) const;	// Compressed HCAL Et
        bool trimmedPlus() const;
        bool trimmedMinus() const;
        int innerEta(  ) const;	// Weighted position eta
        int innerPhi(  ) const;	// Weighted position phi
        L1CaloTowerRef getSeedTower(  ) const;
        const L1CaloTowerRefVector & getConstituents(  ) const;
        L1CaloTowerRef getConstituent( int );
        int seedEmEt() const;
        int seedHadEt() const;
        int constituentEmEt(int , int ) const;
        int constituentHadEt(int , int ) const;
        void shareConstituent(int, int, int);
        L1CaloTowerRef getFriend( int );





        // Electron Variables
        bool fg(  ) const;	// Finegrain bit
        bool eGamma(  ) const;	// Electron/Photon bit
        int eGammaValue(  ) const;	// Electron/Photon bit
		const bool & isoEG(  ) const;	// Egamma Isolatioon

        // Trigger Results
        bool isEGamma(  ) const;	// Returns the EGAMMA decision 
        bool isIsoEGamma(  ) const;	// Returns the iso EGAMMA decision 
        const int &isoEmEtEG(  ) const;
        const int &isoHadEtEG(  ) const;

        const math::PtEtaPhiMLorentzVector & p4(  ) const;	// returns Physics wise LorentzVector in eta,phi continuous space



        private:
        // Refs to the caloTowwers
        L1CaloTowerRef       mSeedTower;
        L1CaloTowerRefVector mConstituents;
        std::vector<int>     mConstituentSharing; 
        L1CaloTowerRefVector mFriends;

        int mSeedThreshold;
        int mEmThreshold;
        int mHadThreshold;// for H/E calculation

        // Coordinates of the reference Point 
        int mIeta;
        int mIphi;
        int mEmEt;
        int mHadEt;

        bool mTrimmedPlus;
        bool mTrimmedMinus;
        bool mFg;
        bool mEgamma;
        int mEgammavalue;
        bool mIsoeg;
        int mIsoEmEtEG; // SH: addition
        int mIsoHadEtEG; // SH: addition
        int mInnereta;
        int mInnerphi;

        math::PtEtaPhiMLorentzVector mP4;	// Lorentz Vector of precise position

    };

}

// Sorting functor
namespace std{
	bool operator< ( const l1slhc::L1CaloClusterWithSeed & aLeft,  const l1slhc::L1CaloClusterWithSeed & aRight );
}

std::ostream & operator<<( std::ostream & , const l1slhc::L1CaloClusterWithSeed & );


#endif
