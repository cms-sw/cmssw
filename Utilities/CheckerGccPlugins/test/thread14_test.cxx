// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
//
// thread14_test: testing check_direct_static_use/check_assign_address_of_static
//                with a const static object
//                with a not_const_thread_safe member.

#pragma ATLAS check_thread_safety

struct S1
{
  S1() : x(0) {}
  void m [[gnu::not_const_thread_safe]] ();
  int x;
};


const static S1 s1;

int f1()
{
  return s1.x;
}

int f2 [[gnu::not_reentrant]] ()
{
  return s1.x;
}

const int* f3()
{
  return &s1.x;
}


const int* f4 [[gnu::not_reentrant]] ()
{
  return &s1.x;
}



struct S2
{
  S2() : x(0) {}
  void m();
  int x;
};

const static S2 s2;


int f11()
{
  return s2.x;
}

const int* f12()
{
  return &s2.x;
}


