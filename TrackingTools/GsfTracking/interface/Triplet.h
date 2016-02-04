#ifndef Triplet_H
#define Triplet_H

/** triplet is identical to stl::pair exceot that it has three members
 *  instead of two (first, second, third).
 */

template <class T1, class T2, class T3>
struct Triplet {
  typedef T1 first_type;
  typedef T2 second_type;
  typedef T3 third_type;

  T1 first;
  T2 second;
  T3 third;
  Triplet() : first(T1()), second(T2()), third(T3()) {}
  Triplet(const T1& a, const T2& b, const T3& c) : 
    first(a), second(b), third(c) {}

  template <class U1, class U2, class U3>
  Triplet(const Triplet<U1, U2, U3>& p) : 
    first(p.first), second(p.second), third(p.third) {}
};

template <class T1, class T2, class T3>
inline bool operator==(const Triplet<T1, T2, T3>& x, 
		       const Triplet<T1, T2, T3>& y) { 
  return x.first == y.first && x.second == y.second && x.third==y.third; 
}

template <class T1, class T2, class T3>
inline bool operator<(const Triplet<T1, T2, T3>& x, 
		      const Triplet<T1, T2, T3>& y) { 
  bool pair_less =  
    x.first < y.first || (!(y.first < x.first) && x.second < y.second); 

  return pair_less || 
    (!(y.first < x.first) && !(y.second < x.second) && x.third < y.third);
}

template <class T1, class T2, class T3>
inline Triplet<T1, T2, T3> make_Triplet(const T1& x, const T2& y, const T3& z) {
  return Triplet<T1, T2, T3>(x, y, z);
}

#endif
