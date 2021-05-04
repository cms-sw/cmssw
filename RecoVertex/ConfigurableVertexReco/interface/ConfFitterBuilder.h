#ifndef _ConfFitterBuilder_H_
#define _ConfFitterBuilder_H_

#include "RecoVertex/ConfigurableVertexReco/interface/VertexFitterManager.h"

/*! @class ConfFitterBuilder
 *  template class that registers an AbstractConfReconstructor
 */

template <class O>
class ConfFitterBuilder {
public:
  ConfFitterBuilder<O>(const std::string& name, const std::string& description) {
    VertexFitterManager::Instance().registerFitter(
        name, []() -> AbstractConfFitter* { return new O(); }, description);
  }
};

#endif  // _ConfFitterBuilder_H_
