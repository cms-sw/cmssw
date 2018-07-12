// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// testing check_direct_static_use

#pragma ATLAS check_thread_safety


int f1(int xx)
{
  static int x;
  x = xx;
  return x;
}

int f2(int xx)
{
  static thread_local int x;
  x = xx;
  return x;
}


int f3(int xx)
{
  static int x [[gnu::thread_safe]];
  // cppcheck-suppress AssignmentIntegerToAddress
  x = xx;
  // cppcheck-suppress CastAddressToIntegerAtReturn
  return x;
}


int f4()
{
  static const int x = 10;
  return x;
}


int f5 [[gnu::not_thread_safe]] (int xx)
{
  static int x;
  return xx + x;
}


int f6 [[gnu::not_reentrant]] (int xx)
{
  static int x;
  return xx + x;
}


struct Y
{
  int y1;
  int y2;
};

static Y yy1;

static int y1;
static thread_local int y2;
static int y3 [[gnu::thread_safe]];
static const int y4 = 0;
static int y5[10];


int g1()
{
  int a = 0;
  a = y1 + y2 + y3 + y4 + yy1.y1 + y5[5];
  return a;
}


void g2(int x)
{
  y1 = x;
  y2 = x;
  // cppcheck-suppress AssignmentIntegerToAddress
  y3 = x;
  yy1.y2 = x;
  y5[4] = x;
}


int g3(int x)
{
  return (x>0) ? y1 : y2;
}


struct H
{
  static int h1;
  static thread_local int h2;
  static int h3 [[gnu::thread_safe]];
  static const int h4 = 0;

  int hee1();
  void hee2(int);
};


int H::hee1()
{
  int a = 0;
  a = h1 + h2 + h3 + h4;
  return a;
}


void H::hee2(int x)
{
  h1 = x;
  h2 = x;
  // cppcheck-suppress AssignmentIntegerToAddress
  h3 = x;
}


struct S1
{
  S1() : x(0) {}
  int x;
};


const static S1 s1;

int h1()
{
  return s1.x;
}

