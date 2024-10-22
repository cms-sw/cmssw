#ifndef _ConfRecoBuilder_H_
#define _ConfRecoBuilder_H_

#include "RecoVertex/ConfigurableVertexReco/interface/VertexRecoManager.h"

/*! @class ConfRecoBuilder
 *  template class that registers an AbstractConfReconstructor
 */

template <class O>
class ConfRecoBuilder {
public:
  ConfRecoBuilder(const std::string& name, const std::string& description) {
    VertexRecoManager::Instance().registerReconstructor(
        name, []() -> AbstractConfReconstructor* { return new O(); }, description);
  }
};

#endif  // _ConfRecoBuilder_H_
