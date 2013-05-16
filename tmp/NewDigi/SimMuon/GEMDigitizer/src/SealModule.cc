#include "SimMuon/GEMDigitizer/src/GEMFactory.h"

#include "SimMuon/GEMDigitizer/src/GEMDigiProducer.h"
#include "SimMuon/GEMDigitizer/src/GEMCSCPadDigiProducer.h"
DEFINE_FWK_MODULE(GEMDigiProducer);
DEFINE_FWK_MODULE(GEMCSCPadDigiProducer);

#include "SimMuon/GEMDigitizer/src/GEMTimingTrivial.h"
#include "SimMuon/GEMDigitizer/src/GEMTimingSimple.h"
#include "SimMuon/GEMDigitizer/src/GEMTimingDetailed.h"
DEFINE_EDM_PLUGIN(GEMTimingFactory, GEMTimingTrivial, "GEMTimingTrivial");
DEFINE_EDM_PLUGIN(GEMTimingFactory, GEMTimingSimple, "GEMTimingSimple");
DEFINE_EDM_PLUGIN(GEMTimingFactory, GEMTimingDetailed, "GEMTimingDetailed");

#include "SimMuon/GEMDigitizer/src/GEMNoiseTrivial.h"
#include "SimMuon/GEMDigitizer/src/GEMNoiseSimple.h"
#include "SimMuon/GEMDigitizer/src/GEMNoiseDetailed.h"
DEFINE_EDM_PLUGIN(GEMNoiseFactory, GEMNoiseTrivial, "GEMNoiseTrivial");
DEFINE_EDM_PLUGIN(GEMNoiseFactory, GEMNoiseSimple, "GEMNoiseSimple");
DEFINE_EDM_PLUGIN(GEMNoiseFactory, GEMNoiseDetailed, "GEMNoiseDetailed");

#include "SimMuon/GEMDigitizer/src/GEMClusteringTrivial.h"
#include "SimMuon/GEMDigitizer/src/GEMClusteringSimple.h"
#include "SimMuon/GEMDigitizer/src/GEMClusteringDetailed.h"
DEFINE_EDM_PLUGIN(GEMClusteringFactory, GEMClusteringTrivial, "GEMClusteringTrivial");
DEFINE_EDM_PLUGIN(GEMClusteringFactory, GEMClusteringSimple, "GEMClusteringSimple");
DEFINE_EDM_PLUGIN(GEMClusteringFactory, GEMClusteringDetailed, "GEMClusteringDetailed");

#include "SimMuon/GEMDigitizer/src/GEMEfficiencyTrivial.h"
#include "SimMuon/GEMDigitizer/src/GEMEfficiencySimple.h"
#include "SimMuon/GEMDigitizer/src/GEMEfficiencyDetailed.h"
DEFINE_EDM_PLUGIN(GEMEfficiencyFactory, GEMEfficiencyTrivial, "GEMEfficiencyTrivial");
DEFINE_EDM_PLUGIN(GEMEfficiencyFactory, GEMEfficiencySimple, "GEMEfficiencySimple");
DEFINE_EDM_PLUGIN(GEMEfficiencyFactory, GEMEfficiencyDetailed, "GEMEfficiencyDetailed");

