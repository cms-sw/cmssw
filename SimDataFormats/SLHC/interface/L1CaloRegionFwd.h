#ifndef L1CaloRegion_Fwd
#define L1CaloRegion_Fwd

#include "SimDataFormats/SLHC/interface/EtaPhiContainer.h"
#include "DataFormats/Common/interface/Ref.h"
#include <vector>


namespace l1slhc
{
	class L1CaloRegion;
}

namespace l1slhc
{
	// typedef std::vector<L1CaloRegion> L1CaloRegionCollection;
	typedef EtaPhiContainer < L1CaloRegion > L1CaloRegionCollection;


	typedef edm::Ref < L1CaloRegionCollection > L1CaloRegionRef;
	typedef std::vector < L1CaloRegionRef > L1CaloRegionRefVector;
}


#endif
