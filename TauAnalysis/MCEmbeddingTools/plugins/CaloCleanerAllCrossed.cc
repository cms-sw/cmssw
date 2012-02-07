#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerAllCrossed.h" 

CaloCleanerAllCrossed::CaloCleanerAllCrossed(const edm::ParameterSet& config)
: CaloCleanerBase(config)
{

}


float CaloCleanerAllCrossed::cleanRH(DetId det, float energy)
{

  if ( h1->find(det.rawId()) != h1->end() || h2->find(det.rawId()) != h2->end()  )
      return 0;

  return energy;

}

