#ifndef UTILITIES_GENERAL_OWN_PTR_H
#define UTILITIES_GENERAL_OWN_PTR_H
//
//  a pointer which "owns" the pointed object
//  similar to std auto_ptr but does allow standard copy constructor
//  that passes ownership when copied
//  lost ownership = zero pointer (safer...)
//

//
//   Version 1.0    V.I.   6/4/99
//   Version 2.0    V.I.   23/4/2001
//   Version 3.0    V.I.   29/1/2004
//     policies introduced...
//

namespace OwnerPolicy {
  struct Transfer{
    template <class X> static X * copy(X * &p) { X* it=p; p=0; return it;} 
    template <class X> static void remove(X * p) { delete p;} 
  };
  struct Copy{
    template <class X> static X * copy(X *p) { return (p) ? new X(*p) : 0;} 
    template <class X> static void remove(X * p) { delete p;} 
  };
  struct Clone{
    template <class X> static X * copy(X *p) { return (p) ? (*p).clone() : 0;} 
    template <class X> static void remove(X * p) { delete p;} 
  };
  struct Replica{
    template <class X> static X * copy(X *p) { return (p) ? (*p).replica() : 0;} 
    template <class X> static void remove(X * p) { delete p;} 
  };
}


/** a pointer which "owns" the pointed object
 */
template <class X, typename P=OwnerPolicy::Transfer> class own_ptr {
private:
  X* ptr;
public:
  typedef X element_type;
  ///
  explicit own_ptr(X* p = 0) : ptr(p) {}
  
  ///
  own_ptr(const own_ptr& a)  : ptr(a.release()) {}
  
#ifndef CMS_NO_TEMPLATE_MEMBERS
  ///
  template <class T, typename Q> own_ptr(const own_ptr<T,Q>& a) 
    : ptr(a.release()) {}
#endif
  ///
  own_ptr& operator=(const own_ptr& a)  {
    if (a.get() != ptr) {
      P::remove(ptr);
      ptr = a.release();
    }
    return *this;
  }

#ifndef CMS_NO_TEMPLATE_MEMBERS
  ///
  template <class T, typename Q> own_ptr& operator=(const own_ptr<T,Q>& a)  {
    if (a.get() != ptr) {
      P::remove(ptr);
      ptr = a.release();
    }
    return *this;
  }
#endif
  ///
  ~own_ptr() {
    P::remove(ptr);
  }
  ///
  X& operator*() const  { return *ptr; }
  ///
  X* operator->() const  { return ptr; }
  ///
  X* get() const  { return ptr; }
  ///
  X* release() const { 
    return P::copy(const_cast<own_ptr<X,P>*>(this)->ptr);
  }

  ///
  void reset(X* p = 0) { 
    if (p != ptr) {
      P::remove(ptr);
      ptr = p;
    }
  }
};

#endif // UTILITIES_GENERAL_OWN_PTR_H
