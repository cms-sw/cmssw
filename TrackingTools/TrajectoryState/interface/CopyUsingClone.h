#ifndef CopyUsingClone_H
#define CopyUsingClone_H

/** A policy definition class for ProxyBase.
 *  The reference counted object is copied using it's clone()
 *  virtual copy constructor. 
 */

template <class T>
class CopyUsingClone {
public:
  T* clone( const T& value) { return value.clone();}
};

#endif // CopyUsingClone_H

