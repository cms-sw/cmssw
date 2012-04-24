#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerAllCrossed.h" 

CaloCleanerAllCrossed::CaloCleanerAllCrossed(const edm::ParameterSet& config)
: CaloCleanerBase(config)
{

}


float CaloCleanerAllCrossed::cleanRH(DetId det, float energy)
{

  // = remove completely
  if ( hPlus->find(det.rawId()) != hPlus->end() || hMinus->find(det.rawId()) != hMinus->end()  )
      return -1;

  // = do nothing with rh
  return 0;

}

