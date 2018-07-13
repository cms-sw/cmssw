// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// thread12_test: testing check_not_const_thread_safe_call

#pragma ATLAS check_thread_safety


struct S
{
  void f1 [[gnu::not_const_thread_safe]] () const;
  void f2() const;
  void f3();
  void f4 [[gnu::not_const_thread_safe]] () const;
  void f5(const S* s) const;
};


void S::f2() const
{
  f1();
}


void S::f3()
{
  f1();
}


void S::f4  [[gnu::not_const_thread_safe]] () const
{
  f1();
}


void S::f5 (const S* s) const
{
  s->f1();
}


