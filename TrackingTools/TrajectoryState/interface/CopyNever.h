#ifndef CopyNever_H
#define CopyNever_H

#include "CommonDet/CDUtilities/interface/DetExceptions.h"

/** A policy definition class for ProxyBase.
 *  The policy does not allow copying of the reference
 *  counted object, and throws an exception if a copy is attempted.
 */

template <class T>
class CopyNever {
public:
  T* clone( const T& value) { 
    throw DetLogicError("Attempt to clone a ProxyBase<CopyNever>");
    return 0;
  }
};

#endif // CopyNever_H

