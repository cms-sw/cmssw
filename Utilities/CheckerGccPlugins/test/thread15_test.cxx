// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
//
// thread15_test: testing check_static_argument_not_const_thread_safe_call:
// a function passing a static const object to an argument_not_const_thread_safe
// function must be not_reentrant.

#pragma ATLAS check_thread_safety

const static int s = 0;

int f1 [[gnu::argument_not_const_thread_safe]] (const int* x);

void f2()
{
  f1 (&s);
  const int* ss = &s;
  f1 (ss);
}

void f3 [[gnu::not_reentrant]] ()
{
  f1 (&s);
}

