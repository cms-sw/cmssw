#ifndef L1CaloRegion_h
#define L1CaloRegion_h

/* This ClassDescribes the 4x4 region

   M.Bachtis, S.Dasu University of Wisconsin-Madison */

namespace l1slhc
{

	class L1CaloRegion
	{
	  public:
		L1CaloRegion(  );
		L1CaloRegion( const int&, const int&, const int& );
		 ~L1CaloRegion(  );

		// Get Functions
		const int& iEta(  ) const;		// Eta of origin in integer coordinates
		const int& iPhi(  ) const;		// Phi of Origin in integer
		const int& E(  ) const;		// Compressed Et 



	  private:
		int mIeta;
		int mIphi;
		int mE;

	};

}


// Sorting functor
namespace std{
	bool operator<( const l1slhc::L1CaloRegion & aLeft,  const l1slhc::L1CaloRegion & aRight );
}

#endif
