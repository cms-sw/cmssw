#include "TrackingTools/TrajectoryState/interface/ChurnAllocator.h"

#include <iostream>

struct A {
  virtual ~A() {}
};
struct B : public A {
  ~B() override { std::cout << "D B " << this << std::endl; }
  explicit B(int) { std::cout << "C B " << this << std::endl; }
};
struct C : public A {
  ~C() override { std::cout << "D c " << this << std::endl; }
  explicit C(int, float) { std::cout << "C B " << this << std::endl; }
};

int main(int k, const char **) {
  using PA = std::shared_ptr<A>;
  //  using PB = std::shared_ptr<B>;
  // using PC = std::shared_ptr<C>;

  PA b = std::allocate_shared<B>(churn_allocator<B>(), 3);
  PA c = std::allocate_shared<C>(churn_allocator<C>(), 3, -2.3);
  std::cout << "more " << std::endl;
  PA b1 = std::allocate_shared<B>(churn_allocator<B>(), 3);
  PA c1 = std::allocate_shared<C>(churn_allocator<C>(), 3, -2.3);
  if (k < 3) {
    b1.reset();
    c.reset();
    std::cout << "churn " << std::endl;
    b1 = std::allocate_shared<B>(churn_allocator<B>(), 3);
    c = std::allocate_shared<C>(churn_allocator<C>(), 3, -2.3);
    b1.reset();
    c.reset();
  }

  std::cout << "end " << std::endl;

  return 0;
}
