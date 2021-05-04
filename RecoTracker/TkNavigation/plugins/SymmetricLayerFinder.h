#ifndef TkNavigation_SymmetricLayerFinder_H
#define TkNavigation_SymmetricLayerFinder_H
#include "FWCore/Utilities/interface/Visibility.h"

#include <vector>
#include <map>

class DetLayer;
class BarrelDetLayer;
class ForwardDetLayer;

/** A symmetrisation algorithm for layer navigation.
 *  For every ForwardDetLayer returns a pointer to the symmetric one
 *  with respect to origin.
 */

class dso_hidden SymmetricLayerFinder {
  typedef std::vector<const ForwardDetLayer*> FDLC;
  typedef FDLC::iterator FDLI;
  typedef FDLC::const_iterator ConstFDLI;
  typedef std::pair<const ForwardDetLayer*, const ForwardDetLayer*> PairType;

public:
  SymmetricLayerFinder(const FDLC&);

  const ForwardDetLayer* mirror(const ForwardDetLayer* layer) { return theForwardMap[layer]; }

  FDLC mirror(const FDLC& input);

private:
  //  typedef map< const ForwardDetLayer*, const ForwardDetLayer*, less<const ForwardDetLayer*> >
  typedef std::map<const ForwardDetLayer*, const ForwardDetLayer*, std::less<const ForwardDetLayer*> > ForwardMapType;

  ForwardMapType theForwardMap;

  const ForwardDetLayer* mirrorPartner(const ForwardDetLayer* layer, const FDLC& rightLayers);
};
#endif  // SymmetricLayerFinder_H
