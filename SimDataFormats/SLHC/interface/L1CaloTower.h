/* L1CaloTower class Keeps both ECAL/HCAL Tower information . Used as input for Clustering

   M.Bachtis,S.Dasu University of Wisconsin-Madison

   Modified Andrew W. Rose Imperial College, London */


#ifndef L1CaloTower_h
#define L1CaloTower_h

#include <vector>


namespace l1slhc
{

	class L1CaloTower
	{

	  public:

		L1CaloTower(  );
		L1CaloTower( const int &, const int & );	// Set Tower position (iEta,iPhi)
		 ~L1CaloTower(  );


		// Setters
		void setPos( const int &, const int & );	// Set Tower position (iEta,iPhi)
		// void setParams( const int& , const int& , const bool& ); //Set ECAL,HCAL Energy and finegrain 

		void setEcal( const int &, const bool & );
		void setHcal( const int &, const bool & );

		// getters 
		const int &E(  ) const;
		const int &H(  ) const;
		const int &iEta(  ) const;
		const int &iPhi(  ) const;
		const bool & EcalFG(  ) const;
		const bool & HcalFG(  ) const;

	  private:
		int mEcal;
		int mHcal;
		int mIeta;
		int mIphi;
		bool mEcalFG;
		bool mHcalFG;


	};

}

// Sorting functor
namespace std{
	bool operator< ( const l1slhc::L1CaloTower & aLeft,  const l1slhc::L1CaloTower & aRight );
}

#endif
