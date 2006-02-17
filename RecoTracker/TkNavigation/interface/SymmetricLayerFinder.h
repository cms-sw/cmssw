#ifndef TkNavigation_SymmetricLayerFinder_H
#define TkNavigation_SymmetricLayerFinder_H

#include <vector>
#include <map>

class DetLayer;
class BarrelDetLayer;
class ForwardDetLayer;

using namespace std;

/** A symmetrisation algorithm for layer navigation.
 *  For every ForwardDetLayer returns a pointer to the symmetric one
 *  with respect to origin.
 */

class SymmetricLayerFinder {

  typedef vector<ForwardDetLayer*>                   FDLC;
  typedef FDLC::iterator                             FDLI;
  typedef FDLC::const_iterator                       ConstFDLI;
  typedef pair< ForwardDetLayer*, ForwardDetLayer*>  PairType;

public:

  SymmetricLayerFinder( const FDLC&);

  ForwardDetLayer* mirror( const ForwardDetLayer* layer) {
    return theForwardMap[layer];
  }

  FDLC mirror( const FDLC& input);

private:

  //  typedef map< const ForwardDetLayer*, const ForwardDetLayer*, less<const ForwardDetLayer*> >
  typedef map< const ForwardDetLayer*, ForwardDetLayer*, less<const ForwardDetLayer*> >
 ForwardMapType;

  ForwardMapType theForwardMap;

  ForwardDetLayer* mirrorPartner( const ForwardDetLayer* layer,
				  const FDLC& rightLayers);
  

};
#endif // SymmetricLayerFinder_H

