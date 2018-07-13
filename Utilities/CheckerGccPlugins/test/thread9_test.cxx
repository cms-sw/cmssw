// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// testing check_returns

#pragma ATLAS check_thread_safety


struct S
{
  int* x;
  int* f1() const;
  int* f2 [[gnu::not_const_thread_safe]] () const;
  const int* f3() const;
  int* f4();
};


int* S::f1() const
{
  int* y = x;
  return y;
}


int* S::f2() const
{
  int* y = x;
  return y;
}


const int* S::f3() const
{
  return x;
}


int* S::f4()
{
  return x;
}


