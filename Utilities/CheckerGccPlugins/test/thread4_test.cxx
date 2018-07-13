// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// testing check_discarded_const

#pragma ATLAS check_thread_safety

const int* xx();


int* f1(const int* y)
{
  return const_cast<int*>(y);
}


int* f2(const int* y)
{
  return (int*)(y);
}


int* f3(const int y[10])
{
  return (int*)(&y[5]);
}

struct S
{
  int x;
  const int y;
};

int* f4(const S& s)
{
  return &const_cast<int&>(s.x);
}

int* f5(S* s)
{
  return (int*)&s->y;
}


int* f6(const int* y)
{
  int* yy [[gnu::thread_safe]] = (int*)y;
  return yy;
}


int* f7(const int* y)
{
  const int* yy = y;
  const int* yyy = yy;
  return const_cast<int*>(yyy);
}

int* f8 [[gnu::argument_not_const_thread_safe]] (const int* y)
{
  const int* yy = y;
  const int* yyy = yy;
  return const_cast<int*>(yyy);
}


int* f8a [[gnu::not_const_thread_safe]] (const int* y)
{
  const int* yy = y;
  const int* yyy = yy;
  return const_cast<int*>(yyy);
}


int* f9  [[gnu::not_const_thread_safe]] ()
{
  const int* y = xx();
  return const_cast<int*>(y);
}


int* f9a  [[gnu::argument_not_const_thread_safe]] ()
{
  const int* y = xx();
  return const_cast<int*>(y);
}


int* f10 [[gnu::not_thread_safe]] (const int* y)
{
  return const_cast<int*>(y);
}


typedef int* vt;
void f11a (const vt& p);
void f11( const int* value )
{
  int* p [[gnu::thread_safe]] = const_cast<int*> (value);
  f11a (p);
}
