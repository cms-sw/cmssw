#ifndef L1CaloCluster_h
#define L1CaloCluster_h


#include <ostream>
/* This ClassDescribes the 2x2 cluster thing 0|1 - - The Cluster reference point is 0 (ieta,iphi)=0,0 2|3

   M.Bachtis, S.Dasu University of Wisconsin-Madison */


//S. Harper changes:
//WARNING: this class contains MANY SUPRISES
//okay this is a temporary solution as we need something to work right now (22/05) and it has been decided to rewrite this class new the near future for a better package
//it has been decided that e/gamma + taus will continue to share the same cluster class rather than having seperate classes
//however they have little incommon with each other than they are both calo objects
//interface changes
//1) need to store emEt and hadEt seperately
//2) need to store emIsolEt and hadIsolEt seperately (this may change but we study them seperately for now. Instead of repurposing the isolation variables, I just added them for now
//3) E is renamed Et() because thats what it is, it also takes an int telling you if its EmEt, HadEt or TotEt to return, this is really handy for the cluster producers and filtering and swapping between tau and e/gamma mode
//4) leadTowerE is always em+had because its a tau thing


#include "DataFormats/Math/interface/LorentzVector.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"

namespace l1slhc
{

	class L1CaloCluster
	{
	        //SH: this operator does not have mIsoenergyeg or tau included for some reason, I have note updated it to have mIsoEmEtEG or mIsoHadEtEG either
		friend bool operator==( const l1slhc::L1CaloCluster & a, const l1slhc::L1CaloCluster & b )
		{
			if ( a.mIeta != b.mIeta )
				return false;
			if ( a.mIphi != b.mIphi )
				return false;
			if ( a.mFg != b.mFg )
				return false;
			if ( a.mEgamma != b.mEgamma )
				return false;
			if ( a.mLeadtowertau != b.mLeadtowertau )
				return false;
			if ( a.mEgammavalue != b.mEgammavalue )
				return false;
			if ( a.mInnereta != b.mInnereta )
				return false;
			if ( a.mInnerphi != b.mInnerphi )
				return false;
			if ( a.mIsoclusterseg != b.mIsoclusterseg )
				return false;
			if ( a.mIsoclusterstau != b.mIsoclusterstau )
				return false;
			if ( a.mIsoeg != b.mIsoeg )
				return false;
			if ( a.mIsotau != b.mIsotau )
				return false;
			if ( a.mCentral != b.mCentral )
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
		  L1CaloCluster(  );
		  L1CaloCluster( const int &, const int & );
		 ~L1CaloCluster(  );

		void setFg( const bool & );	// Set FG Bit
		void setEGamma( const bool & );	// Set EGamma Bit
 		void setEGammaValue( const int & );	// Set EGamma Value E/E+H (%)
		void setIsoClusters( const int &, const int & );	// Number of isolated objectsisolation Clusters
		void setIsoEnergy( const int &, const int & );	// Energy of Isolation
	        void setIsoEmAndHadEtEG( const int &, const int & );
		void setIsoEG( const bool & );	// EG isolation
		void setIsoTau( const bool & );	// Tau isolation
		void setCentral( const bool & );	// Central Bit 
		void setLeadTower( const bool & );	// Lead Tower over threshold bit for taus
		void setLorentzVector( const math::PtEtaPhiMLorentzVector & );	// Central Bit 
		void setPosBits( const int &, const int & );
		void setConstituents( const L1CaloTowerRefVector & );	
                void setEmEt( const int & );	
                void setHadEt( const int & );
		void setLeadTowerE( const int & );
		void setSecondTowerE( const int & );
		void setThirdTowerE( const int & );
		void setFourthTowerE( const int & );
		void setRing1E( const int & );
		void setRing2E( const int & );
		void setRing3E( const int & );
		void setRing4E( const int & );
		void addConstituent( const L1CaloTowerRef & );
	        int hasConstituent( const int &, const int & )const; //SH change: should be const so adding const
		void removeConstituent( const int &, const int & );

		// Get Functions
		const int &iEta(  ) const;	// Eta of origin in integer coordinates
		const int &iPhi(  ) const;	// Phi of Origin in integer
		const int Et(int mode=0x3) const;	// Compressed Et SH: name change  mode = bit 1 add EmEt, bit 2 add HadEt, I had to make this a non-reference variable as it returns the sum of mEmEt + mHadEt, leave as is for now
	        const int &EmEt()const;	// SH: addition
	        const int &HadEt()const; // SH: addition
		const int &LeadTowerE(  ) const;	// Lead Tower Et 
		const int &SecondTowerE(  ) const;	// Lead Tower Et 
		const int &ThirdTowerE(  ) const;	// Lead Tower Et 
		const int &FourthTowerE(  ) const;	// Lead Tower Et 
		const int &Ring1E(  ) const;	// Lead Tower Et 
		const int &Ring2E(  ) const;	// Lead Tower Et 
		const int &Ring3E(  ) const;	// Lead Tower Et 
		const int &Ring4E(  ) const;	// Lead Tower Et 
		const int &innerEta(  ) const;	// Weighted position eta
		const int &innerPhi(  ) const;	// Weighted position phi
		const L1CaloTowerRefVector & getConstituents(  ) const;
	        L1CaloTowerRef getConstituent( const int & )const; //SH change: should be const so adding const





		// Electron Variables
		const bool & fg(  ) const;	// Finegrain bit
		const bool & eGamma(  ) const;	// Electron/Photon bit
		const int &eGammaValue(  ) const;	// Electron/Photon bit

		// isolation Variables
		const bool & isCentral(  ) const;	// Means that the cluster was not pruned during isolation
		const bool & isoEG(  ) const;	// Egamma Isolatioon
		const bool & isoTau(  ) const;	// Tau isolation
		const int &isoClustersEG(  ) const;	// 2x2 isolation clusters for Egamma cuts
		const int &isoClustersTau(  ) const;	// 2x2 isolation clusters for Tau Cut
		const int &isoEnergyEG(  ) const;	// 2x2 isolation clusters for Egamma cuts
		const int &isoEnergyTau(  ) const;	// 2x2 isolation clusters for Egamma cuts
		const bool & hasLeadTower(  ) const;

	        const int &isoEmEtEG(  ) const;
	        const int &isoHadEtEG(  ) const;
                  
		// Trigger Results
		bool isEGamma(  ) const;	// Returns the EGAMMA decision 
		bool isIsoEGamma(  ) const;	// Returns the iso EGAMMA decision 
		bool isTau(  ) const;	// returns The Tau decison
		bool isIsoTau(  ) const;	// returns The Tau decison

		const math::PtEtaPhiMLorentzVector & p4(  ) const;	// returns Physics wise LorentzVector in eta,phi continuous space



	  private:
		// Refs to teh caloTowwers
		L1CaloTowerRefVector mConstituents;

		// Coordinates of the reference Point 
		int mIeta;
		int mIphi;
		int mEmEt;
	        int mHadEt;
		int mLeadTowerE; //Lead Tower Energy
		int mSecondTowerE; //Lead Tower Energy
		int mThirdTowerE; //Lead Tower Energy
		int mFourthTowerE; //Lead Tower Energy
		int mRing1E; //Lead Tower Energy
		int mRing2E; //Lead Tower Energy
		int mRing3E; //Lead Tower Energy
		int mRing4E; //Lead Tower Energy
		// FineGrain / EGamma /Isolations

		bool mFg;
		bool mEgamma;
		bool mCentral;
		bool mIsoeg;
		bool mLeadtowertau;
		bool mIsotau;
		int mEgammavalue;
		int mInnereta;
		int mInnerphi;
		int mIsoclusterseg;
	        int mIsoenergyeg;
		int mIsoenergytau;
		int mIsoclusterstau;

		math::PtEtaPhiMLorentzVector mP4;	// Lorentz Vector of precise position

                int mIsoEmEtEG; // SH: addition
	        int mIsoHadEtEG; // SH: addition
	  
	};

}

// Sorting functor
namespace std{
	bool operator< ( const l1slhc::L1CaloCluster & aLeft,  const l1slhc::L1CaloCluster & aRight );
}

std::ostream & operator<<( std::ostream & , const l1slhc::L1CaloCluster & );


#endif
