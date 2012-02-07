#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerMVA.h" 

CaloCleanerMVA::CaloCleanerMVA(const edm::ParameterSet& config)
: CaloCleanerBase(config)
{

}




float CaloCleanerMVA::cleanRH(DetId det, float energy){
  throw "Implement me!\n";
  return 0.;
}


