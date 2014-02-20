#ifndef L1CaloClusterWithSeed_Fwd
#define L1CaloClusterWithSeed_Fwd


#include "SimDataFormats/SLHC/interface/EtaPhiContainer.h"
#include "DataFormats/Common/interface/Ref.h"

namespace l1slhc
{
	class L1CaloClusterWithSeed;
}

namespace l1slhc
{
	// typedef std::vector<L1CaloClusterWithSeed> L1CaloClusterWithSeedCollection;
	typedef EtaPhiContainer < L1CaloClusterWithSeed > L1CaloClusterWithSeedCollection;

	typedef edm::Ref < L1CaloClusterWithSeedCollection > L1CaloClusterWithSeedRef;
}

#endif
