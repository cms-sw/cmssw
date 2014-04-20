#include <memory>
#include <iostream>

template <typename T>
class churn_allocator: public std::allocator<T>
{
public:
  using Base = std::allocator<T>;
  using pointer = typename Base::pointer;
  using size_type = typename Base::size_type;

  static pointer&  cache() {
    static thread_local pointer local = nullptr;
    return local;
  } 
  static bool & gard() {
    static thread_local bool g = false;
    return g;
  }


  template<typename _Tp1>
  struct rebind
  {
    typedef churn_allocator<_Tp1> other;
  };

  pointer allocate(size_type n, const void *hint=0)
  {
    if (!gard()) 
     cache() = std::allocator<T>::allocate(n, hint);
    gard()=false; return cache();
  }
  
  void deallocate(pointer p, size_type n)
  {
    if (p==cache()) gard()=true;
    else std::allocator<T>::deallocate(p, n);
  }
  
  churn_allocator() = default;
  churn_allocator(churn_allocator const&)=default;
  churn_allocator(churn_allocator &&)=default;

  template <class U>                    
  churn_allocator(const churn_allocator<U> &a) noexcept: std::allocator<T>(a) { }
};


struct A {virtual ~A(){} };
struct B :public A{ ~B(){std::cout << "D B " << this << std::endl;} explicit B(int){std::cout << "C B " << this << std::endl;} };
struct C :public A{ ~C(){std::cout << "D c " << this << std::endl;} explicit C(int,float){std::cout << "C B " << this << std::endl;}  };


int main(int k, const char **) {

  using PA = std::shared_ptr<A>;
  //  using PB = std::shared_ptr<B>;
  // using PC = std::shared_ptr<C>;
  
  PA b = std::allocate_shared<B>(churn_allocator<B>(),3);
  PA c = std::allocate_shared<C>(churn_allocator<C>(),3,-2.3);
  std::cout << "more " << std::endl;
  PA b1 = std::allocate_shared<B>(churn_allocator<B>(),3);
  PA c1 = std::allocate_shared<C>(churn_allocator<C>(),3,-2.3);
  if (k<3) {
    b1.reset();
    c.reset();
    std::cout << "churn " << std::endl;
    b1 = std::allocate_shared<B>(churn_allocator<B>(),3);
    c = std::allocate_shared<C>(churn_allocator<C>(),3,-2.3);
    b1.reset();
    c.reset();
  }

  std::cout << "end " << std::endl;


  return 0;
}
