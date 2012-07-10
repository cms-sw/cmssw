#ifndef L1TowerJet_h
#define L1TowerJet_h

/* 
   This class describves the L1 Reconstructed jet M.Bachtis,S.Dasu University of Wisconsin - Madison */

#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"


namespace l1slhc
{

	class L1TowerJet
	{

	  public:

		enum tJetShape
		{
			square,
			circle
		};

	  public:

		L1TowerJet( );
		L1TowerJet( const int& , const L1TowerJet::tJetShape& );
		L1TowerJet( const int& , const L1TowerJet::tJetShape& , const int& , const int& );
		 ~L1TowerJet(  );

		// getters
		const int& iEta(  ) const;
		const int& iPhi(  ) const;
		const int& E(  ) const;
		const bool& central(  ) const;
		const math::PtEtaPhiMLorentzVector& p4(  ) const;	// returns LorentzVector in eta,phi space
		const int& JetSize(  ) const;
		const L1TowerJet::tJetShape& JetShape(  ) const;

//possibly helpful methods
		double EcalVariance(  ) const;
		double HcalVariance(  ) const;
		double EnergyVariance(  ) const;

		// Setters

		void setP4( const math::PtEtaPhiMLorentzVector & );
		void setCentral( const bool& );
//		void setE( const int& );


		void addConstituent( const L1CaloTowerRef & );
		L1CaloTowerRefVector::iterator getConstituent( const int&, const int& );
		void removeConstituent( const int&, const int& );
		
		const L1CaloTowerRefVector& getConstituents(  ) const;



	  private:
		int mIeta;
		int mIphi;
		int mE;
		bool mCentral;

		int mJetSize;
		L1TowerJet::tJetShape mJetShapeType;



		L1CaloTowerRefVector mConstituents;
		math::PtEtaPhiMLorentzVector mP4;

	};


}


// Sorting functor
namespace std{
	bool operator< ( const l1slhc::L1TowerJet & aLeft,  const l1slhc::L1TowerJet & aRight );
}


std::ostream & operator<<( std::ostream & , const l1slhc::L1TowerJet & );

#endif
