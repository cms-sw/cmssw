#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerConst.h" 
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <boost/foreach.hpp>

CaloCleanerConst::CaloCleanerConst(const edm::ParameterSet& config)
: CaloCleanerBase(config)
{

  std::vector<std::string> names = config.getParameter < std::vector<std::string> >("names");
  BOOST_FOREACH (std::string & name, names){
     values_[name] = config.getParameter <double>(name);
  }


}




float CaloCleanerConst::cleanRH(DetId det, float energy){

  float correction =0;

  int num = 0;
  if ( hPlus->find(det.rawId()) != hPlus->end() )   { ++num; }
  if ( hMinus->find(det.rawId()) != hMinus->end() ) { ++num; }
  if ( num == 0) return 0; // not crossed by mu.

  std::string key = detNaming_.getKey(det);
  if (values_.find(key) == values_.end()) {
      throw cms::Exception("CaloCleanerConst") << "Deep trouble. I know nothing about " << key;
  }

  correction += num*values_[key];

  //std::cout <<  key << " " << energy << " " << correction  << std::endl;
  if (correction > energy ) correction = -1; // remove rh completely
  return correction;

}


