// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// thread10_test: testing check_thread_safe_call

#include <string.h>
void* operator new(size_t, void*) noexcept;
void* operator new[](size_t) noexcept;

void f1 [[gnu::check_thread_safety]] ();
void f2 [[gnu::not_thread_safe]] ();
void f3 ();

namespace std {
void foo();
struct Bar { static void bar(); };
}
namespace boost { void btest(); }
struct SS {};

void f4 [[gnu::check_thread_safety]] (char* s)
{
  f1();
  f2();
  f3();
  strlen(s);
  strtok(s, ".");
  char* saveptr = nullptr;
  strtok_r(s, ".", &saveptr);
  std::foo();
  std::Bar::bar();
  boost::btest();
  new(s) SS;
  new char[sizeof(int)];
}


struct TBuffer {};
TBuffer& operator>> (TBuffer&, int&);
TBuffer& operator<< (TBuffer&, int);

int f5 [[gnu::check_thread_safety]] (TBuffer& b, int x)
{
  b << x;
  int x2;
  b >> x2;
  return x2;
}

