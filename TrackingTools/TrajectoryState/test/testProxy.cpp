#include "TrackingTools/TrajectoryState/interface/ProxyBase11.h"

#include <iostream>

struct A {
  virtual ~A() {}
  virtual std::shared_ptr<A> clone() const = 0;

  template <typename T, typename... Args>
  static std::shared_ptr<A> build(Args &&...args) {
    return std::allocate_shared<T>(std::allocator<T>(), std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  static std::shared_ptr<A> churn(Args &&...args) {
    return std::allocate_shared<T>(churn_allocator<T>(), std::forward<Args>(args)...);
  }
};
template <class T>
struct ACloned : public A {
  std::shared_ptr<A> clone() const { return std::allocate_shared<T>(std::allocator<T>(), *this); }
};
struct B final : public A {
  ~B() { std::cout << "D B " << this << std::endl; }
  explicit B(int) { std::cout << "C B " << this << std::endl; }
  std::shared_ptr<A> clone() const { return build<B>(*this); }
};
struct C final : public A {
  ~C() { std::cout << "D C " << this << std::endl; }
  explicit C(int, float) { std::cout << "C C " << this << std::endl; }
  std::shared_ptr<A> clone() const { return build<C>(*this); }
};

struct BB final : public A {
  ~BB() { std::cout << "D BB " << this << std::endl; }
  explicit BB(int) { std::cout << "C BB " << this << std::endl; }
  std::shared_ptr<A> clone() const { return churn<BB>(*this); }
};
struct CC final : public A {
  ~CC() { std::cout << "D CC " << this << std::endl; }
  explicit CC(int, float) { std::cout << "C CC " << this << std::endl; }
  std::shared_ptr<A> clone() const { return churn<CC>(*this); }
};

using Proxy = ProxyBase11<A>;

void astd(int k) {
  std::cout << "\nstd allocator\n" << std::endl;

  using PA = Proxy;
  //  using PB = std::shared_ptr<B>;
  // using PC = std::shared_ptr<C>;

  PA b = A::build<B>(3);
  PA c = A::build<C>(3, -2.3);
  std::cout << "more " << std::endl;
  PA b1 = A::build<B>(3);
  PA c1 = A::build<C>(3, -2.3);
  if (k < 3) {
    b1.reset();
    c.reset();
    std::cout << "churn " << std::endl;
    b1 = b.data().clone();
    c = c1.data().clone();
    b1.reset();
    c.reset();
  }

  std::cout << b.references() << ' ' << &b.data() << ' ';
  std::cout << &b.unsharedData() << std::endl;
  c = b;
  std::cout << b.references() << ' ' << &b.data() << ' ';
  std::cout << &b.unsharedData() << std::endl;
  std::cout << b.references() << ' ' << &b.data() << ' ';
  std::cout << &b.unsharedData() << std::endl;
  std::cout << c.references() << ' ' << &c.data() << ' ';
  std::cout << &c.unsharedData() << std::endl;

  std::cout << "end " << std::endl;
}

void achurn(int k) {
  std::cout << "\nchurn allocator\n" << std::endl;

  using PA = Proxy;
  //  using PB = std::shared_ptr<B>;
  // using PC = std::shared_ptr<C>;

  PA b = A::churn<BB>(3);
  PA c = A::churn<CC>(3, -2.3);
  std::cout << "more " << std::endl;
  PA b1 = A::churn<BB>(3);
  PA c1 = A::churn<CC>(3, -2.3);
  if (k < 3) {
    b1.reset();
    c.reset();
    std::cout << "churn " << std::endl;
    b1 = b.data().clone();
    c = c1.data().clone();
    b1.reset();
    c.reset();
  }

  std::cout << b.references() << ' ' << &b.data() << ' ';
  std::cout << &b.unsharedData() << std::endl;
  c = b;
  std::cout << b.references() << ' ' << &b.data() << ' ';
  std::cout << &b.unsharedData() << std::endl;
  std::cout << b.references() << ' ' << &b.data() << ' ';
  std::cout << &b.unsharedData() << std::endl;
  std::cout << c.references() << ' ' << &c.data() << ' ';
  std::cout << &c.unsharedData() << std::endl;

  std::cout << "end " << std::endl;
}

int main(int k, const char **) {
  astd(k);
  achurn(k);

  return 0;
}
