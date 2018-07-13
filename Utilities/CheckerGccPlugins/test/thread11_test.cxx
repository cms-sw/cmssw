// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// thread11_test: testing check_not_reentrant_call

#pragma ATLAS check_thread_safety


void f1();
void f2 [[gnu::not_reentrant]] ();

void f3()
{
  f1();
  f2();
}

void f4 [[gnu::not_reentrant]] ()
{
  f1();
  f2();
}
