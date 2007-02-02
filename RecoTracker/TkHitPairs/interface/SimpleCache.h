#ifndef TkHitPairs_SimpleCache_H
#define TkHitPairs_SimpleCache_H

/** \class TkHitPairs::SimpleCache
 * Stores ValueType* value for a given KeyType key. 
 * The value is deleted during clearing
 */

#include <vector>

namespace TkHitPairs {

template <class KeyType, class ValueType> class SimpleCache {
public:
  SimpleCache(int initSize) { reserve(initSize); }

  virtual ~SimpleCache() { clear(); }

  void reserve(int size) { theContainer.reserve(size); }

  /// get object associated to Key. If not found 0 returned
  const ValueType*  get(const KeyType & key) {
    for (ConstItr it = theContainer.begin(); it != theContainer.end(); it++) {
      if ( it->first == key) return it->second;
    }
    return 0;
  }

  /// add object to cache. It is caller responsibility to check that object
  /// is not yet in cache.
  void add(const KeyType & key, const ValueType * value) {
    theContainer.push_back( std::make_pair(key,value));
  }

  /// emptify cache, delete values associated to Key
  virtual void clear() {
    for (ConstItr i=theContainer.begin(); i!= theContainer.end(); i++) {
      delete i->second;
    }
    theContainer.clear();
  }

protected:
  typedef std::pair< KeyType, const ValueType * > KeyValuePair;
  std::vector< KeyValuePair > theContainer;
  typedef typename std::vector< KeyValuePair >::const_iterator ConstItr;

private:
  SimpleCache(const SimpleCache &) { }

};
}
#endif

