#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerMVA.h" 
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <boost/foreach.hpp>

CaloCleanerMVA::CaloCleanerMVA(const edm::ParameterSet& config)
: CaloCleanerBase(config)
{

  std::vector<std::string> names = config.getParameter < std::vector<std::string> >("names");
  BOOST_FOREACH (std::string & name, names){

    TMVA::Reader * reader = new TMVA::Reader("!Color:Silent");

    reader->AddVariable("pt",&pt);
    reader->AddVariable("p",&p);
    reader->AddVariable("eta",&eta);
    reader->AddVariable("phi",&phi);
    reader->AddVariable("len",&len);
    reader->AddVariable("charge",&charge);


    std::vector<std::string> params = config.getParameter < std::vector<std::string> >(name);
    edm::FileInPath Weigths("TauAnalysis/MCEmbeddingTools/data/"+params.at(1));
    reader->BookMVA( params.at(0), Weigths.fullPath().c_str()  );    

    readers_[name] = reader;
    methods_[name] = params.at(0);
  }


}




float CaloCleanerMVA::cleanRH(DetId det, float energy){

  float correction =0;

  //std::cout << "Called for " << det.rawId() << " " << energy << std::endl;
  //std::cout << hPlus->size() << " " << hMinus->size() << std::endl;
  if ( ZmumuHandle_->size()==0){
    std::cout << "Empty input col!\n" << std::endl;
    return 0;
  }

  std::vector<size_t> todo;
  if ( hPlus->find(det.rawId()) != hPlus->end() ) {
    std::cout << "BBB" << std::endl;
    if (ZmumuHandle_->at(0).daughter(0)->charge() > 0) todo.push_back(0);
    if (ZmumuHandle_->at(0).daughter(1)->charge() > 0) todo.push_back(1);
  }

  if ( hMinus->find(det.rawId()) != hMinus->end())  {
    std::cout << "AAA" << std::endl;
    if (ZmumuHandle_->at(0).daughter(0)->charge() < 0) todo.push_back(0);
    if (ZmumuHandle_->at(0).daughter(1)->charge() < 0) todo.push_back(1);
  }

  if ( todo.size() == 0) return 0;

  std::cout << " XXX "  << std::endl;

  for (size_t in=0; in < todo.size(); ++in ){
    len =-1; charge = -999; pt = -1; p = -1; eta = -999; phi = -999;

    charge = ZmumuHandle_->at(0).daughter(todo[in])->charge();
    pt = ZmumuHandle_->at(0).daughter(todo[in])->pt();
    p = ZmumuHandle_->at(0).daughter(todo[in])->p();
    eta = ZmumuHandle_->at(0).daughter(todo[in])->eta();
    phi = ZmumuHandle_->at(0).daughter(todo[in])->phi();

    if ( ZmumuHandle_->at(0).daughter(todo[in])->charge() > 0   ){
      if ( hPlus->find(det.rawId())== hPlus->end()  ) 
        throw cms::Exception("CaloCleanerMVA") << "Expected rawId not found in map "; // be extra safe
      len = hPlus->find(det.rawId())->second;
    } else {
      if ( hMinus->find(det.rawId())== hPlus->end()  ) 
        throw cms::Exception("CaloCleanerMVA") << "Expected rawId not found in map "; // be extra safe
      len = hMinus->find(det.rawId())->second;
    }
    std::string key = detNaming_.getKey(det);
    if (readers_.find(key) == readers_.end()) {
      throw cms::Exception("CaloCleanerMVA") << "Deep trouble. I know nothing about " << key;
    }

    float corr = readers_[key]->EvaluateMVA(methods_[key]);
    std::cout << charge << " pt=" << pt << " p=" << p 
              << " eta=" << eta << " phi=" << phi 
              << " len=" << len << " corr=" << corr << std::endl;

    correction += corr;
  }

  std::cout <<  detNaming_.getKey(det) << " " << energy << " " << correction  << std::endl;
  if (correction > energy ) correction = -1; // remove rh completely
  return correction;

}


