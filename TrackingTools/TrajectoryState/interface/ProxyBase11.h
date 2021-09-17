#ifndef Tracker_ProxyBase11_H
#define Tracker_ProxyBase11_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "FWCore/Utilities/interface/Likely.h"

#ifdef TR_DEBUG
#include <iostream>
#endif

#include "ChurnAllocator.h"

/** A base class for reference counting proxies.
 *  The class to which this one is proxy must inherit from
 *  ReferenceCounted.
 *  A ProxyBase has value semantics, in contrast to ReferenceCountingPointer,
 *  which has pointer semantics.
 *  The Proxy class inheriting from ProxyBase must duplicate the
 *  part of the interface of the reference counted class that it whiches to expose.
 */

template <class T>
class ProxyBase11 {
public:
  using pointer = std::shared_ptr<T>;

  // protected:

  ProxyBase11() {}

  explicit ProxyBase11(T* p) : theData(p) {}
  template <typename U>
  ProxyBase11(std::shared_ptr<U> p) : theData(std::move(p)) {}
  template <typename U>
  ProxyBase11& operator=(std::shared_ptr<U> p) {
    theData = std::move(p);
    return *this;
  }

  ~ProxyBase11() noexcept { destroy(); }

  void swap(ProxyBase11& other) noexcept { std::swap(theData, other.theData); }

  ProxyBase11(ProxyBase11&& other) noexcept = default;
  ProxyBase11& operator=(ProxyBase11&& other) noexcept = default;
  ProxyBase11(ProxyBase11 const& other) = default;
  ProxyBase11& operator=(const ProxyBase11& other) = default;

  void reset() { theData.reset(); }

  const T& data() const {
    check();
    return *theData;
  }

  T& unsharedData() {
    check();
    if (references() > 1) {
      theData = theData->clone();
    }
    return *theData;
  }

  T& sharedData() {
    check();
    return *theData;
  }

  bool isValid() const { return bool(theData); }

  void check() const {
#ifdef TR_DEBUG
    if UNLIKELY (!theData)
      std::cout << "dead proxyBase11 " << references() << std::endl;
#endif
  }

  void destroy() noexcept {}
  int references() const { return theData.use_count(); }

private:
  std::shared_ptr<T> theData;
};

template <class T>
inline void swap(ProxyBase11<T>& lh, ProxyBase11<T>& rh) noexcept {
  lh.swap(rh);
}

#endif  // Tracker_ProxyBase11_H
