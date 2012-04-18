#ifndef L1CaloTower_Fwd
#define L1CaloTower_Fwd

#include "DataFormats/Common/interface/Ref.h"
#include "SimDataFormats/SLHC/interface/EtaPhiContainer.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"

#include <vector>

/* 
   namespace l1slhc { class L1CaloTower; } */


namespace l1slhc
{


	// typedef std::vector<L1CaloTower> L1CaloTowerCollection;
	typedef EtaPhiContainer < L1CaloTower > L1CaloTowerCollection;

	typedef edm::Ref < L1CaloTowerCollection > L1CaloTowerRef;

	typedef std::vector < L1CaloTowerRef > L1CaloTowerRefVector;
}

#endif
