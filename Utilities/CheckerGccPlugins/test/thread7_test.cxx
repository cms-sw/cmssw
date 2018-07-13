// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// testing check for static/mutable members

#pragma ATLAS check_thread_safety
namespace std {
class mutex{};
template <class T> struct atomic {
  void store(T);
};
}
struct SS {};


struct S
{
  S() : a(), b(), c() {}
  mutable int a;
  mutable int b [[gnu::thread_safe]];
  mutable int *c[10];
  mutable std::mutex m;
  typedef std::mutex mutex_t;
  mutable mutex_t m2;
  typedef std::atomic<int> atomic_t;
  mutable atomic_t a1;

  static int d;
  static int e [[gnu::thread_safe]];
  static std::atomic<int> f;
  static std::atomic<SS> g;
  static atomic_t f1;
  static const int h;
  static thread_local int i;
};


template <class T>
struct X
{
  X() {}
  mutable int a;
};
