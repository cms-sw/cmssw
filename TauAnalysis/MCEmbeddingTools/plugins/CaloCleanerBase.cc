#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerBase.h" 

CaloCleanerBase::CaloCleanerBase(const edm::ParameterSet& config){

     colPlus = config.getParameter<edm::InputTag>("depsPlus");
     colMinus = config.getParameter<edm::InputTag>("depsMinus");

     muons_ = config.getParameter<edm::InputTag>("ZmumuCands");
}

void CaloCleanerBase::setEvent(edm::Event& iEvent) { 

   //std::cout << "CaloCleanerBase::setEvent" << std::endl;
   iEvent_ = &iEvent; 
   if (!iEvent_->getByLabel(colPlus, hPlus))
     std::cout << "Cannot get: " << colPlus << std::endl;

   if (!iEvent_->getByLabel(colMinus, hMinus))
     std::cout << "Cannot get: " << colMinus << std::endl;

   //std::cout << "XXX pm sizes: " << hPlus->size() << " " << hMinus->size() << std::endl;

   bool mumuOK = iEvent_->getByLabel(muons_, ZmumuHandle_);

   if (!mumuOK) std::cout << "XXX Cannot find Zmumu! " << std::endl;
   if (ZmumuHandle_->size() == 0) std::cout << "XXX Zmumu collection empty! " << std::endl;

}
