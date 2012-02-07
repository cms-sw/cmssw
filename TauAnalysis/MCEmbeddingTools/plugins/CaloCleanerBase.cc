#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerBase.h" 

CaloCleanerBase::CaloCleanerBase(const edm::ParameterSet& config){

     col1 = config.getParameter<edm::InputTag>("depsPlus");
     col2 = config.getParameter<edm::InputTag>("depsMinus");

}

void CaloCleanerBase::setEvent(edm::Event& iEvent) { 

   iEvent_ = &iEvent; 
   iEvent_->getByLabel(col1, h1);
   iEvent_->getByLabel(col2, h2);

}
