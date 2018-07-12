// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// testing check_assign_address_of_static

#pragma ATLAS check_thread_safety


static int y1;
static const int y2 = 0;
//static int y3[10];
static thread_local int y4;
static int y5 [[gnu::thread_safe]];


int* f1()
{
  return &y1;
}


const int* f2()
{
  return &y1;
}


int& f3()
{
  return y1;
}


const int& f4()
{
  return y1;
}


#if 0
int f5()
{
  // warning is different btn 6.2 and 7.
  int* p = &(y3[0]) + 3;
  return *p;
}


int f6()
{
  // warning is different btn 6.2 and 7.
  const int* p = &(y3[0]) + 3;
  return *p;
}
#endif


const int* f7()
{
  return &y2;
}


struct Y
{
  int y1;
  int y2;
  void foo();
};


int* f8 [[gnu::not_reentrant]] ()
{
  return &y1;
}


int* f9()
{
  return &y4;
}


int* f10()
{
  return &y5;
}


int* f11 [[gnu::not_thread_safe]] ()
{
  return &y1;
}


class Foo
{
public:
  char* p;
  Foo();
};

const Foo y12[1];

Foo f12(int i)
{
  return y12[i];
}
