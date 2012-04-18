/* SLHC Calo Trigger Class for Trigger configuration...Contains wiring and Cuts M.Bachtis,S.Dasu. University of Wisconsin-Madison */


#ifndef L1CaloTriggerSetup_h
#define L1CaloTriggerSetup_h

#include <vector>
#include <map>
#include <string>


class L1CaloTriggerSetup
{
  private:
	int mLatticedim;			// Eta Dimension of the square lattice 
	int mLatticeeta0;			// Eta of the Square Lattice
	int mLatticephi0;			// Phi of the Square Lattice
	int mLatticeetam;			// Eta of the Square Lattice
	int mLatticephim;			// Phi of the Square Lattice
	int mEcalactivitycut;		// Ecal Activity Cut
	int mHcalactivitycut;		// hcalActivity Cut
/* 
	int mElectroncuta; //Electron ID Cut
	int mElectroncutb; //Electron ID Cut
	int mElectroncutc; //Electron ID Cut
*/
	std::vector < int > mElectroncut;
	int mTauseedtower;			// Electron ID Cut
	int mClustercut;			// Cluster Threshold Cut

/* 
	int mIsolationea; //Isolation ratio Electron;
	int mIsolationeb; //Isolation ratio Electron;
*/
	std::vector < int > mIsolatione;
/*
	int mIsolationta;//Isolation ratio Tau;
	int mIsolationtb;//Isolation ratio Tau;
*/
	std::vector < int > mIsolationt;
/* 
	int mIsolationthreg; //Isolation threshold EG;
	int mIsolationthrtau;//Isolation threshold Tau;
*/
	std::vector < int > mIsothr;
	int mIsolationzone;			// Number of towers that define the isolation zone;
	// int mJetcenter ; //jet Center Deviation
	int mJetet;					// jet Center Deviation
	int mFinegrainpass;			// ignore fine grain bit (set it to 0)

	// Geometry Mapping between towers/lattice 
	std::map < int, std::pair < int, int > > mGeomap;


  public:

	const std::pair < int, int > &getTowerEtaPhi( const int &bin ) const
	{
		return mGeomap.find( bin )->second;
	}


	// Lattice Navigation helper Functions
	int getEta( const int &bin ) const	// get the ieta of a specific Bin
	{
		return bin % mLatticedim;

	}
	int getPhi( const int &bin ) const	// get the iphi of a specific bin
	{
		return bin / mLatticedim;
	}

	int getBin( const int &eta, const int &phi ) const	// get the bin for a ieta,iphi pair
	{
		return phi * mLatticedim + eta;
	}



	const int &etaMin(  ) const
	{
		return mLatticeeta0;
	}

	const int &etaMax(  ) const
	{
		return mLatticeetam;
	}

	const int &phiMin(  ) const
	{
		return mLatticephi0;
	}

	const int &phiMax(  ) const
	{
		return mLatticephim;
	}


	const int &ecalActivityThr(  ) const
	{
		return mEcalactivitycut;
	}

	const int &hcalActivityThr(  ) const
	{
		return mHcalactivitycut;
	}

	const int &clusterThr(  ) const
	{
		return mClustercut;
	}

	const int &seedTowerThr(  ) const
	{
		return mTauseedtower;
	}




	const int &nIsoTowers(  ) const
	{
		return mIsolationzone;
	}

/*
	const int& jetCenterDev()
	{
		return mJetcenter;
	}
*/

	const int &minJetET(  ) const
	{
		return mJetet;
	}

	const int &fineGrainPass(  ) const
	{
		return mFinegrainpass;
	}





/* 
	std::vector<int> electronThr()
	{
		std::vector<int> a;
		a.push_back(mElectroncuta);
		a.push_back(mElectroncutb);
		a.push_back(mElectroncutc);
	   return a; 
	}
*/
	const int &electronThr( const int &aIndex ) const
	{
		return mElectroncut.at( aIndex );
	}


/* 
	std::vector<int> isoThr()
	{	
		std::vector<int> a;
		a.push_back(mIsolationthreg);
		a.push_back(mIsolationthrtau);
	   return a;
	}

	std::vector<int> isolationE() {
		std::vector<int> a;
		a.push_back(mIsolationea);
		a.push_back(mIsolationeb);
	   return a;
	}

	std::vector<int> isolationT() {
		std::vector<int> a;
		a.push_back(mIsolationta);
		a.push_back(mIsolationtb);
		return a;
	}
*/

	const int &isoThr( const int &aIndex ) const
	{
		return mIsothr.at( aIndex );
	}

	const int &isolationE( const int &aIndex ) const
	{
		return mIsolatione.at( aIndex );
	}

	const int &isolationT( const int &aIndex ) const
	{
		return mIsolationt.at( aIndex );
	}



	L1CaloTriggerSetup(  ):
	mLatticedim( 1 ), 
	mLatticeeta0( 1 ), 
	mLatticephi0( 1 ), 
	mLatticeetam( -1000 ), 
	mLatticephim( -1111 ), 
	mEcalactivitycut( 2 ), 
	mHcalactivitycut( 6 ),
/*	mElectroncuta(8),
	mElectroncutb(0),
	mElectroncutc(0), */
	mElectroncut( std::vector < int >( 3, 0 ) ),
	mTauseedtower( 0 ), 
	mClustercut( 4 ), 
	mIsolatione( std::vector < int >( 2, 0 ) ), 
	mIsolationt( std::vector < int >( 2, 0 ) ), 
	mIsothr( std::vector < int >( 2, 0 ) )
	{
		mElectroncut.at( 0 ) = 8;
	}

	 ~L1CaloTriggerSetup(  )
	{
	}

	void setGeometry( const int &eta0, const int &phi0, const int &etam, const int &phim, const int &dim )
	{
		mLatticedim = dim;
		mLatticeeta0 = eta0;
		mLatticephi0 = phi0;
		mLatticeetam = etam;
		mLatticephim = phim;
	}

	void addWire( const int &no, const int &eta, const int &phi )	// Configure Wire Connection
	{
		mGeomap[no] = std::make_pair( eta, phi );
	}


	void setThresholds( const int &ecal_a_c, const int &hcal_a_c, const int &egammaA, const int &egammaB, const int &egammaC, const int &tauSeed, const int &clusterCut, const int &isoRatioEA,
						const int &isoRatioEB, const int &isoRatioTA, const int &isoRatioTB, const int &isoZone, const int &isoThresEG, const int &isoThresTau, const int &jetet, const int &fgp )
	{

		mEcalactivitycut = ecal_a_c;
		mHcalactivitycut = hcal_a_c;
/*
		mElectroncuta = egammaA; 
		mElectroncutb = egammaB; 
		mElectroncutc = egammaC; 
*/
		mElectroncut.at( 0 ) = egammaA;
		mElectroncut.at( 1 ) = egammaB;
		mElectroncut.at( 2 ) = egammaC;
		mTauseedtower = tauSeed;
		mClustercut = clusterCut;
/*
		mIsolationea = isoRatioEA;
		mIsolationeb = isoRatioEB;
*/
		mIsolatione.at( 0 ) = isoRatioEA;
		mIsolatione.at( 1 ) = isoRatioEB;
/*
		mIsolationta = isoRatioTA;
		mIsolationtb = isoRatioTB; 
*/
		mIsolationt.at( 0 ) = isoRatioTA;
		mIsolationt.at( 1 ) = isoRatioTB;
		mIsolationzone = isoZone;
/*
		mIsolationthreg = isoThresEG; 
		mIsolationthrtau = isoThresTau;
*/
		mIsothr.at( 0 ) = isoThresEG;
		mIsothr.at( 1 ) = isoThresTau;
		// mJetcenter = jetc;
		mJetet = jetet;
		mFinegrainpass = fgp;
	}


};

#endif
