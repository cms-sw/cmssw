#ifndef _VertexRecoManager_H_
#define _VertexRecoManager_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfReconstructor.h"
#include <map>
#include <string>

/*! \class VertexRecoManager
 *  Class that manages the vertex reconstruction strategies
 */

class VertexRecoManager {

public:
  static VertexRecoManager & Instance();
  void registerReconstructor ( const std::string & name, AbstractConfReconstructor * o,
                  const std::string & description );
  std::string describe ( const std::string & );

  AbstractConfReconstructor * get ( const std::string & );
  std::map < std::string, AbstractConfReconstructor * > get ();

  ~VertexRecoManager();
  VertexRecoManager * clone() const;

private:
  VertexRecoManager ( const VertexRecoManager & );
  VertexRecoManager ();
  std::map < std::string, AbstractConfReconstructor * > theAbstractConfReconstructors;
  std::map < std::string, std::string > theDescription;
};

#endif // _VertexRecoManager_H_
