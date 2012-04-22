#ifndef L1TowerJet_Fwd
#define L1TowerJet_Fwd


#include "SimDataFormats/SLHC/interface/EtaPhiContainer.h"
#include "DataFormats/Common/interface/Ref.h"
#include <vector>

namespace l1slhc
{
	class L1TowerJet;
}

namespace l1slhc
{
	typedef EtaPhiContainer < L1TowerJet > L1TowerJetCollection;

	typedef edm::Ref < L1TowerJetCollection > L1TowerJetRef;
}

#endif
