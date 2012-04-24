#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerNone.h" 

CaloCleanerNone::CaloCleanerNone(const edm::ParameterSet& config)
: CaloCleanerBase(config)
{

}


float CaloCleanerNone::cleanRH(DetId det, float energy)
{


  // = do nothing with rh
  return 0;

}

