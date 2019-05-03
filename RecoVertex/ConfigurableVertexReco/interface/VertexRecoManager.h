#ifndef _VertexRecoManager_H_
#define _VertexRecoManager_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfReconstructor.h"
#include <functional>
#include <map>
#include <memory>
#include <string>

/*! \class VertexRecoManager
 *  Class that manages the vertex reconstruction strategies
 */

class VertexRecoManager {

public:
  static VertexRecoManager &Instance();
  void registerReconstructor(const std::string &name,
                             std::function<AbstractConfReconstructor *()> o,
                             const std::string &description);
  std::string describe(const std::string &) const;

  std::unique_ptr<AbstractConfReconstructor> get(const std::string &) const;
  std::vector<std::string> getNames() const;

  ~VertexRecoManager();
  VertexRecoManager *clone() const;

private:
  VertexRecoManager(const VertexRecoManager &);
  VertexRecoManager();
  std::map<std::string, std::function<AbstractConfReconstructor *()>>
      theAbstractConfReconstructors;
  std::map<std::string, std::string> theDescription;
};

#endif // _VertexRecoManager_H_
