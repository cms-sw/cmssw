#ifndef Tracker_ChurnAllocator_H
#define Tracker_ChurnAllocator_H
#include <memory>

template <typename T>
class churn_allocator : public std::allocator<T> {
public:
  using Base = std::allocator<T>;
  using pointer = typename std::allocator_traits<std::allocator<T>>::pointer;
  using size_type = typename Base::size_type;

  struct Cache {
    pointer cache = nullptr;
    bool gard = false;
  };

  static Cache &cache() {
    static thread_local Cache local;
    return local;
  }

  template <typename _Tp1>
  struct rebind {
    typedef churn_allocator<_Tp1> other;
  };

  pointer allocate(size_type n) {
    Cache &c = cache();
    if (!c.gard)
      c.cache = std::allocator<T>::allocate(n);
    c.gard = false;
    return c.cache;
  }

  void deallocate(pointer p, size_type n) {
    Cache &c = cache();
    if (p == c.cache)
      c.gard = true;
    else
      std::allocator<T>::deallocate(p, n);
  }

  churn_allocator() = default;
  churn_allocator(churn_allocator const &) = default;
  churn_allocator(churn_allocator &&) = default;

  template <class U>
  churn_allocator(const churn_allocator<U> &a) noexcept : std::allocator<T>(a) {}
};

#endif
