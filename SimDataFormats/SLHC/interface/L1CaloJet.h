#ifndef L1CaloJet_h
#define L1CaloJet_h

/* 
   This class describves the L1 Reconstructed jet M.Bachtis,S.Dasu University of Wisconsin - Madison */

#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>
#include "SimDataFormats/SLHC/interface/L1CaloRegionFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegion.h"

namespace l1slhc
{

	class L1CaloJet
	{

	  public:

		L1CaloJet(  );
		L1CaloJet( const int&, const int& );
		 ~L1CaloJet(  );

		// getters
		const int& iEta(  ) const;
		const int& iPhi(  ) const;
		const int& E(  ) const;
		const bool& central(  ) const;
		const math::PtEtaPhiMLorentzVector& p4(  ) const;	// returns LorentzVector in eta,phi space

		// Setters
		void setP4( const math::PtEtaPhiMLorentzVector & );
		void setCentral( const bool& );
		void setE( const int& );

		void addConstituent( const L1CaloRegionRef & );
		int hasConstituent( const int&, const int& );
		void removeConstituent( const int&, const int& );
		
		const L1CaloRegionRefVector& getConstituents(  ) const;



	  private:
		int mIeta;
		int mIphi;
		int mE;
		bool mCentral;

		L1CaloRegionRefVector mConstituents;
		math::PtEtaPhiMLorentzVector mP4;

	};


}


// Sorting functor
namespace std{
	bool operator< ( const l1slhc::L1CaloJet & aLeft,  const l1slhc::L1CaloJet & aRight );
}


std::ostream & operator<<( std::ostream & , const l1slhc::L1CaloJet & );

#endif
