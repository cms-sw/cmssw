#ifndef L1CaloCluster_h
#define L1CaloCluster_h


#include <ostream>
/* This ClassDescribes the 2x2 cluster thing 0|1 - - The Cluster reference point is 0 (ieta,iphi)=0,0 2|3

   M.Bachtis, S.Dasu University of Wisconsin-Madison */

#include "DataFormats/Math/interface/LorentzVector.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"

namespace l1slhc
{

	class L1CaloCluster
	{

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
			if ( a.mE != b.mE )
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
		void setIsoEG( const bool & );	// EG isolation
		void setIsoTau( const bool & );	// Tau isolation
		void setCentral( const bool & );	// Central Bit 
		void setLeadTower( const bool & );	// Lead Tower over threshold bit for taus
		void setLorentzVector( const math::PtEtaPhiMLorentzVector & );	// Central Bit 
		void setPosBits( const int &, const int & );
		void setConstituents( const L1CaloTowerRefVector & );
		void setE( const int & );
		void setLeadTowerE( const int & );
		void addConstituent( const L1CaloTowerRef & );
		int hasConstituent( const int &, const int & );
		void removeConstituent( const int &, const int & );

		// Get Functions
		const int &iEta(  ) const;	// Eta of origin in integer coordinates
		const int &iPhi(  ) const;	// Phi of Origin in integer
		const int &E(  ) const;	// Compressed Et 
		const int &LeadTowerE(  ) const;	// Lead Tower Et 
		const int &innerEta(  ) const;	// Weighted position eta
		const int &innerPhi(  ) const;	// Weighted position phi
		const L1CaloTowerRefVector & getConstituents(  ) const;
		L1CaloTowerRef getConstituent( const int & );





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
		int mE;
		int mLeadTowerE; //Lead Tower Energy
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

	};

}

// Sorting functor
namespace std{
	bool operator< ( const l1slhc::L1CaloCluster & aLeft,  const l1slhc::L1CaloCluster & aRight );
}

std::ostream & operator<<( std::ostream & , const l1slhc::L1CaloCluster & );


#endif
