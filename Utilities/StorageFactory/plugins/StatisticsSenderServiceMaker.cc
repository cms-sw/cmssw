
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "Utilities/StorageFactory/interface/StatisticsSenderService.h"

using edm::storage::StatisticsSenderService;

typedef edm::serviceregistry::AllArgsMaker<StatisticsSenderService> StatisticsSenderServiceMaker;
DEFINE_FWK_SERVICE_MAKER(StatisticsSenderService, StatisticsSenderServiceMaker);

