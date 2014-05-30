#ifndef L1TowerJet_h
#define L1TowerJet_h

/* 
   This class describves the L1 Reconstructed jet M.Bachtis,S.Dasu University of Wisconsin - Madison */

#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/TriggerTowerGeometry.h"

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
		//		L1TowerJet( const int& aJetSize, const L1TowerJet::tJetShape& aJetShapeType , const int& aJetArea , const int& iEta, const int& iPhi);
		L1TowerJet( const int& aJetSize, const L1TowerJet::tJetShape& aJetShapeType , 
			    const std::vector< std::pair< int, int > >& aJetShapeMap, const int &iEta, const int &iPhi  );

		~L1TowerJet(  );

		// getters
		// ~~~~~~~
		// Jet iEta and iPhi parameters (Eta and phi of top-left reference TT)
		const int& iEta(  ) const;
		const int& iPhi(  ) const;
		// Total TT energy sum (GeV)
		const double& E(  ) const;
		const bool& central(  ) const;

		// centrality member variables
		const double& Centrality( ) const;

		//asymmetry member variables
		const int& AsymEta(  ) const;
		const int& AsymPhi(  ) const;


		/* //weighted iEta, iPhi
                const double& iWeightedEta( ) const;
                const double& iWeightedPhi( ) const;
		*/

		// Jet pT
                const double Pt( ) const;
		// Geometric jet center eta and phi
                const double Eta( ) const;
                const double Phi( ) const;
		// Energy weighted eta and phi of jet center
                const double WeightedEta( ) const;
                const double WeightedPhi( ) const;
		// Jet pT, weighted eta, weighted phi and mass
		const math::PtEtaPhiMLorentzVector& p4(  ) const;
		// Jet shape width in TTs
		const int& JetSize(  ) const;
		// Jet shape type
		const L1TowerJet::tJetShape& JetShape(  ) const;
		// Area of jet in TTs
                const int& JetArea(  ) const;
		// Real area of jet (deltaEta * deltaPhi)
		const double& JetRealArea( ) const;


/* 		const double& MaxTTEnergy( ) const; */
/* 		const double& MaxTTEnergyInSeedRegion( ) const; */

//              possibly helpful methods
//		double EcalVariance(  ) const;
//		double HcalVariance(  ) const;
//		double EnergyVariance(  ) const;
		// Median absolute deviations
		double EcalMAD() const;
		double HcalMAD() const;
		double EnergyMAD() const;

		// Setters
		// ~~~~~~~
		void setP4( const math::PtEtaPhiMLorentzVector & p4 );
		void setPt( const double & pT );
		void setCentral( const bool& );

		// Calculate the jet centrality
		void calculateCentrality();
		// Determine the central jet eta and phi for unweighted and energy weighting
		void calculateJetCenter();
		void calculateWeightedJetCenter();

		// Add a TT to the TowerJet
		void addConstituent( const L1CaloTowerRef & Tower );
		L1CaloTowerRefVector::iterator getConstituent( const int& eta , const int& phi );
		void removeConstituent( const int& eta , const int& phi );
		
		const L1CaloTowerRefVector& getConstituents(  ) const;
		
		double MAD( std::deque<int>& aDataSet ) const;


	  private:
		// i-coordinates, define the top left TT of the jet
		int mIeta;
		int mIphi;
		// Energy of jet in 2 GeV units: 1 -> 2 GeV (REMOVE THIS)
		int mE2GeV;
		// True value of E in GeV
		double mE;

/* 		int mMaxTTEnergy2GeV; */
/*                 int mMaxTTEnergyInSeedRegion2GeV;  */

		bool mCentral;

                // Asymmetry parameters
		int mAsymEta;
		int mAsymPhi;
		
		// Centrality parameter
		double mCentrality;

                //weighted eta and phi
		/*double mWeightedIeta;
		double mWeightedIphi;
		*/
		/*
		double mWeightedEta;
		double mWeightedPhi;
		*/
		// Jet center eta and phi
		double mJetCenterEta;
		double mJetCenterPhi;


		// Size and area of jet in TTs 
		int mJetSize;
		L1TowerJet::tJetShape mJetShapeType;
		int mJetArea;
		// Actual eta*phi jet area
		double mJetRealArea;

		// Undo wrapping of iPhi 
		int iPhiUnwrap(int aIphi);

		// Tower geometry converter
		static TriggerTowerGeometry mTowerGeo;

		L1CaloTowerRefVector mConstituents;
		
		// Pt, energy weighted eta, energy weighted phi and mass of the jet
		math::PtEtaPhiMLorentzVector mP4;




		static const double PI;

	};


}


// Jet rank operator
namespace std{
    bool operator<( const l1slhc::L1TowerJet & aLeft, const l1slhc::L1TowerJet & aRight );
    bool operator>( const l1slhc::L1TowerJet & aLeft, const l1slhc::L1TowerJet & aRight );
}

// Jet printing
std::ostream & operator<<( std::ostream & , const l1slhc::L1TowerJet & );

#endif
