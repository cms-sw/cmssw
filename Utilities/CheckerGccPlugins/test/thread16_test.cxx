// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// thread16_test: testing check_virtual_overrides

#pragma ATLAS check_thread_safety

struct B
{
  virtual void f1 [[gnu::not_reentrant]] ();
  virtual void f2();
  virtual void f3();
};


struct D : public B
{
  virtual void f1();
  virtual void f2 [[gnu::not_reentrant]] ();
  virtual void f3 [[gnu::not_const_thread_safe]] ();
};
