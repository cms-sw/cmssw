#ifndef Tracker_ProxyBase11_H
#define Tracker_ProxyBase11_H

#include "TrackingTools/TrajectoryState/interface/CopyUsingNew.h"
#include "FWCore/Utilities/interface/Visibility.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include <memory>

/** A base class for reference counting proxies.
 *  The class to which this one is proxy must inherit from
 *  ReferenceCounted.
 *  A ProxyBase has value semantics, in contrast to ReferenceCountingPointer,
 *  which has pointer semantics.
 *  The Proxy class inheriting from ProxyBase must duplicate the
 *  part of the interface of the reference counted class that it whiches to expose.
 */

template <class T, class Cloner > 
class ProxyBase11 {
public:


protected:

  ProxyBase11() {}

  explicit ProxyBase11( T* p) theData(p) {}

  ~ProxyBase11()  noexcept {
    destroy();
  }

  void swap(ProxyBase11& other)  noexcept {
    std::swap(theData,other.theData);
  }

#ifdef CMS_NOCXX11
  ProxyBase11& operator=( const ProxyBase11& other) {
    return *this;
  }
  ProxyBase11( const ProxyBase11& other) {}
#else
  ProxyBase11(ProxyBase11&& other)  noexcept = default;
  ProxyBase11& operator=(ProxyBase11&& other)  noexcept = default; 
  ProxyBase11(ProxyBase11 const & other) = default;
  ProxyBase11& operator=( const ProxyBase11& other) = default;
#endif

  const T& data() const { check(); return *theData;}

  T& unsharedData() {
    check(); 
    if ( references() > 1) {
      theData.reset(Cloner().clone(*theData));
    }
    return *theData;
  }

  T& sharedData() { check(); return *theData;}

  bool isValid() const { return theData;}

  void check() const {
#ifdef TR_DEBUG
    if  unlikely(!theData)
      throw TrajectoryStateException("Error: uninitialized ProxyBase11 used");
#endif
  }

  void destroy()  noexcept {}

  int  references() const {return theData->count();}  

private:
#ifdef CMS_NOCXX11
  T *  theData;
#else
  std::shared_ptr<T> theData;
#endif
};

template <class T, class Cloner >
inline
void swap(ProxyBase11<T,Cloner>& lh, ProxyBase11<T,Cloner>& rh)  noexcept {
  lh.swap(rh);
}

#endif // Tracker_ProxyBase11_H
