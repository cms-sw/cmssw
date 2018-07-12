// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// thread13_test: testing check_argument_not_const_thread_safe_call

#pragma ATLAS check_thread_safety


void f1 [[gnu::argument_not_const_thread_safe]] (const int*);
void f2 [[gnu::argument_not_const_thread_safe]] (int);

void f10 (int* x)
{
  int y;
  int* xp = x+10;
  f1(x);
  f1(&y);
  f1(xp);
  f1(x+20);
  f2(*x);
}

void f11 [[gnu::argument_not_const_thread_safe]] (int* x)
{
  f1(x);
}


void f12 (int* x[10])
{
  f1(x[4]);
}


struct S
{
  void m1() const;
  void m2 [[gnu::argument_not_const_thread_safe]] () const;
  void m3();
  int* x;
  int yy[10];

  struct SS
  {
    int y[10];
  };
  SS ss;
};


void S::m1() const
{
  f1(x);
  f1(&ss.y[5]);
  f1(&yy[5]);
}


void S::m2 [[gnu::argument_not_const_thread_safe]] () const
{
  f1(x);
}


void S::m3()
{
  f1(x);
}


