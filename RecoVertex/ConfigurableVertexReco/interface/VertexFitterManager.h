#ifndef _VertexFitterManager_H_
#define _VertexFitterManager_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfFitter.h"
#include <map>
#include <string>

/*! \class VertexFitterManager
 *  Class that manages the vertex reconstruction strategies
 */

class VertexFitterManager {

public:
  static VertexFitterManager & Instance();
  void registerFitter ( const std::string & name, AbstractConfFitter * o,
                          const std::string & description );
  std::string describe ( const std::string & );

  AbstractConfFitter * get ( const std::string & );
  std::map < std::string, AbstractConfFitter * > get ();

  ~VertexFitterManager();
  VertexFitterManager * clone() const;

private:
  VertexFitterManager ( const VertexFitterManager & );
  VertexFitterManager ();
  std::map < std::string, AbstractConfFitter * > theAbstractConfFitters;
  std::map < std::string, std::string > theDescription;
};

#endif // _VertexFitterManager_H_
