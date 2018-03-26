
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

//#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBTrigPrimTestAlgo.h"
//#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBTrigPrimClusterAlgo.h"

#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBTPCluster.h"
#include <SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBTPCluster.h>

#include <vector>


#ifdef _CINT_
#pragma link C++ class EcalEBTPCluster;
#pragma link C++ class std::vector<EcalEBTPCluster>;
#endif


namespace {
	struct dictionary {
		EcalEBTPCluster dummy_EcalEBTPCluster;
		std::vector<EcalEBTPCluster> dummy_EcalEBTPCluster_vec;
	};
}

