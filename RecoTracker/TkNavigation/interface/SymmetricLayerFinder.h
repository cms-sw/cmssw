#ifndef TkNavigation_SymmetricLayerFinder_H
#define TkNavigation_SymmetricLayerFinder_H

#include <vector>
#include <map>

class DetLayer;
class BarrelDetLayer;
class ForwardDetLayer;

/** A symmetrisation algorithm for layer navigation.
 *  For every ForwardDetLayer returns a pointer to the symmetric one
 *  with respect to origin.
 */

class SymmetricLayerFinder {

  typedef std::vector<ForwardDetLayer*>                   FDLC;
  typedef FDLC::iterator                                  FDLI;
  typedef FDLC::const_iterator                            ConstFDLI;
  typedef std::pair< ForwardDetLayer*, ForwardDetLayer*>  PairType;

public:

  SymmetricLayerFinder( const FDLC&);

  ForwardDetLayer* mirror( const ForwardDetLayer* layer) {
    return theForwardMap[layer];
  }

  FDLC mirror( const FDLC& input);

private:

  //  typedef map< const ForwardDetLayer*, const ForwardDetLayer*, less<const ForwardDetLayer*> >
  typedef std::map< const ForwardDetLayer*, ForwardDetLayer*, std::less<const ForwardDetLayer*> >
    ForwardMapType;

  ForwardMapType theForwardMap;

  ForwardDetLayer* mirrorPartner( const ForwardDetLayer* layer,
				  const FDLC& rightLayers);
  

};
#endif // SymmetricLayerFinder_H

