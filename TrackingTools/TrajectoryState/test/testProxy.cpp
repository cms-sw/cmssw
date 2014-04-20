#include "TrackingTools/TrajectoryState/interface/ProxyBase11.h"


#include<iostream>

struct A {virtual ~A(){} virtual std::shared_ptr<A> clone() const =0;};
template <class T> struct ACloned : public A  { std::shared_ptr<A> clone() const { return std::allocate_shared<T>(std::allocator<T>(),*this);} };
struct B final : public A { 
  ~B(){std::cout << "D B " << this << std::endl;} explicit B(int){std::cout << "C B " << this << std::endl;} 
  std::shared_ptr<A> clone() const { return std::allocate_shared<B>(std::allocator<B>(),*this); }
};
struct C final :public A  { 
  ~C(){std::cout << "D c " << this << std::endl;} explicit C(int,float){std::cout << "C B " << this << std::endl;} 
  std::shared_ptr<A> clone() const { return std::allocate_shared<C>(std::allocator<C>(),*this);}
};

using Proxy = ProxyBase11<A>;


int main(int k, const char **) {

  using PA = Proxy;
  //  using PB = std::shared_ptr<B>;
  // using PC = std::shared_ptr<C>;
  
  PA b = std::allocate_shared<B>(std::allocator<B>(),3);
  PA c = std::allocate_shared<C>(std::allocator<C>(),3,-2.3);
  std::cout << "more " << std::endl;
  PA b1 = std::allocate_shared<B>(std::allocator<B>(),3);
  PA c1 = std::allocate_shared<C>(std::allocator<C>(),3,-2.3);
  if (k<3) {
    b1.reset();
    c.reset();
    std::cout << "churn " << std::endl;
    b1 = b.data().clone();
    c =  c1.data().clone();
    b1.reset();
    c.reset();
  }

  std::cout << b.references() << ' ' << &b.data() << ' ' << &b.unsharedData() << std::endl;
  c=b;
  std::cout << b.references() << ' ' << &b.data() << ' ';
  std::cout << &b.unsharedData() << std::endl;
  

  std::cout << "end " << std::endl;
  
  
  return 0;
}

