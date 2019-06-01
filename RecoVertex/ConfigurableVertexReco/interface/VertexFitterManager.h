#ifndef _VertexFitterManager_H_
#define _VertexFitterManager_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfFitter.h"
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <memory>

/*! \class VertexFitterManager
 *  Class that manages the vertex reconstruction strategies
 */

class VertexFitterManager {
public:
  static VertexFitterManager& Instance();
  void registerFitter(const std::string& name, std::function<AbstractConfFitter*()> o, const std::string& description);
  std::string describe(const std::string&) const;

  std::unique_ptr<AbstractConfFitter> get(const std::string&) const;
  std::vector<std::string> getNames() const;

  ~VertexFitterManager();
  VertexFitterManager* clone() const;

private:
  VertexFitterManager(const VertexFitterManager&);
  VertexFitterManager();
  std::map<std::string, std::function<AbstractConfFitter*()> > theAbstractConfFitters;
  std::map<std::string, std::string> theDescription;
};

#endif  // _VertexFitterManager_H_
