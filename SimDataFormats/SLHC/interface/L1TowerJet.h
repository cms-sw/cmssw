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
		L1TowerJet( const int& aJetSize, const L1TowerJet::tJetShape& aJetShapeType , const int& aJetArea );
		L1TowerJet( const int& aJetSize, const L1TowerJet::tJetShape& aJetShapeType , const int& aJetArea , const int& iEta, const int& iPhi);
		 ~L1TowerJet(  );

		// getters
		const int& iEta(  ) const;
		const int& iPhi(  ) const;
		const int& E(  ) const;
		const bool& central(  ) const;

		//asymmetry member variables
		const int& AsymEta(  ) const;
		const int& AsymPhi(  ) const;

                //weighted iEta, iPhi
                const double& iWeightedEta( ) const;
                const double& iWeightedPhi( ) const;

                const double& WeightedEta( ) const;
                const double& WeightedPhi( ) const;

		const math::PtEtaPhiMLorentzVector& p4(  ) const;	// returns LorentzVector in eta,phi space
		const int& JetSize(  ) const;
		const L1TowerJet::tJetShape& JetShape(  ) const;

//              possibly helpful methods
//		double EcalVariance(  ) const;
//		double HcalVariance(  ) const;
//		double EnergyVariance(  ) const;

		double EcalMAD() const;
		double HcalMAD() const;
		double EnergyMAD() const;

		// Setters

		void setP4( const math::PtEtaPhiMLorentzVector & p4 );
		void setCentral( const bool& );
//		void setE( const int& );

                void CalcWeightediEta();
                void CalcWeightediPhi();

		void calculateWeightedEta( );
		void calculateWeightedPhi();




		void addConstituent( const L1CaloTowerRef & Tower );
		L1CaloTowerRefVector::iterator getConstituent( const int& eta , const int& phi );
		void removeConstituent( const int& eta , const int& phi );
		
		const L1CaloTowerRefVector& getConstituents(  ) const;



	  private:
		int mIeta;
		int mIphi;
		int mE;
		bool mCentral;

                //add asym
		int mAsymEta;
		int mAsymPhi;

                //weighted eta and phi
		double mWeightedIeta;
		double mWeightedIphi;
		double mWeightedEta;
		double mWeightedPhi;


		int mJetSize;
		L1TowerJet::tJetShape mJetShapeType;
		int mJetArea;


		L1CaloTowerRefVector mConstituents;
		math::PtEtaPhiMLorentzVector mP4;


		double MAD( std::deque<int>& aDataSet ) const;

	};


}


// Sorting functor
namespace std{
	bool operator< ( const l1slhc::L1TowerJet & aLeft,  const l1slhc::L1TowerJet & aRight );
}


std::ostream & operator<<( std::ostream & , const l1slhc::L1TowerJet & );

#endif
