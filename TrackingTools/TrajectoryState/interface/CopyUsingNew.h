#ifndef CopyUsingNew_H
#define CopyUsingNew_H

/** A policy definition class for ProxyBase.
 *  The reference counted object is copied using "new".
 */

template <class T>
class CopyUsingNew {
public:
  T* clone( const T& value) { return new T(value);}
};

#endif // CopyUsingNew_H

