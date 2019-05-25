#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "RecoVertex/ConfigurableVertexReco/interface/VertexRecoManager.h"

using namespace std;

void VertexRecoManager::registerReconstructor(const string& name,
                                              std::function<AbstractConfReconstructor*()> o,
                                              const string& d) {
  theAbstractConfReconstructors[name] = o;
  theDescription[name] = d;
}

VertexRecoManager::~VertexRecoManager() {}

std::string VertexRecoManager::describe(const std::string& d) const {
  auto found = theDescription.find(d);
  if (found == theDescription.end()) {
    return std::string();
  }
  return found->second;
}

VertexRecoManager* VertexRecoManager::clone() const { return new VertexRecoManager(*this); }

VertexRecoManager::VertexRecoManager(const VertexRecoManager& o) {
  std::cout << "[VertexRecoManager] copy constructor! Error!" << std::endl;
  exit(0);
  /*
  for ( map < string, AbstractConfReconstructor * >::const_iterator i=o.theAbstractConfReconstructors.begin(); 
        i!=o.theAbstractConfReconstructors.end() ; ++i )
  {
    theAbstractConfReconstructors[ i->first ] = i->second->clone();
  }
  
  theIsEnabled=o.theIsEnabled;
  */
}

VertexRecoManager& VertexRecoManager::Instance() {
  //The singleton's internal structure only changes while
  // this library is being loaded. All other methods are const.
  CMS_THREAD_SAFE static VertexRecoManager singleton;
  return singleton;
}

std::unique_ptr<AbstractConfReconstructor> VertexRecoManager::get(const string& s) const {
  auto found = theAbstractConfReconstructors.find(s);
  if (found == theAbstractConfReconstructors.end()) {
    return std::unique_ptr<AbstractConfReconstructor>{};
  }
  return std::unique_ptr<AbstractConfReconstructor>{found->second()};
}

std::vector<std::string> VertexRecoManager::getNames() const {
  std::vector<std::string> ret;
  ret.reserve(theAbstractConfReconstructors.size());
  for (const auto& i : theAbstractConfReconstructors) {
    ret.push_back(i.first);
  }
  return ret;
}

VertexRecoManager::VertexRecoManager() {}
