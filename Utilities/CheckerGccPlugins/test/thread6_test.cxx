// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// testing check_mutable

#pragma ATLAS check_thread_safety
namespace std {
class mutex{};
template <class T> struct atomic {
  void store(T);
};
}
struct lock_guard { lock_guard (std::mutex&); };


struct B
{
  int z;
};


struct S
{
  mutable int x;
  mutable B b;
  mutable B bb[5];
  mutable int w [[gnu::thread_safe]];
  mutable std::mutex m;
  mutable std::atomic<int> a;
  void f1(int y) const;
  void f6(int y) const;
  void f9(int y) const;
  void f12(int y);
  void f15 [[gnu::not_thread_safe]] (int y) const;
  void f16() const;
  void f17() const;
};


void S::f1(int y) const
{
  x = y;
}


void f2(const S* s)
{
  s->x = 10;
}


void f3(int* x);
void f4(const S* s)
{
  f3(&s->x);
}

int* f5(const S* s)
{
  return &s->x;
}


void S::f6(int y) const
{
  b.z = y;
}

int* f7(const S* s)
{
  return &s->b.z;
}

int* f8(const S* s)
{
  return &s->bb[2].z;
}


void S::f9(int y) const
{
  bb[0].z = y;
}


void f10(const S* s)
{
  s->w = 10;
}


int* f11(const S* s)
{
  return &s->w;
}


void S::f12(int y)
{
  x = y;
}


void f14(S* s)
{
  s->x = 10;
}


void S::f15(int y) const
{
  x = y;
}


void S::f16() const
{
  lock_guard l (m);
}


void S::f17() const
{
  a.store(1);
}
