// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

#include <iostream>


class c
{
public:
  int aa;
private:
  int a;
  int m_b;
  static int s_j;
  static int m_k;
  int s_l;
  static int m;
  static const int ALLCAPS;
  static const int ALL_CAPS123;
  static const int NotAllCaps;
  int* p_x;
  int  p_y;
  static void* fgIsA;
  void m_foo();
};


class TFoo
{
private:
  int fThing;
};


namespace boost {

class Foo
{
private:
  int thing;
};

}


int foo (int c, int m_d, int s_e, int _t)
{
  int x = c + m_d + s_e;
  int m_y
    = 2;
  int s_q = 3;
  static int s_r = 10;
  int _s = 20;
  return x + m_y + s_q + s_r + _s + _t;
}


int m_z;


namespace SG { struct IAuxStore { IAuxStore(); } ; }

class AuxTest
  : public SG::IAuxStore
{
private:
  int foo;
};


class AuxTest2
  : public AuxTest
{
private:
  int bar;
};


template <class T> class ServiceHandle {};
template <class T> class ToolHandle {};

class UseHandle
{
private:
  ServiceHandle<int> p_h;
  ToolHandle<int> p_i;
};


class Foo_p12
{
private:
  int x;
};


// Test for a crash seen.
class Virt
{
  virtual ~Virt();
};



///////////////////////////
// Testing another crash.
class vector
{
public:
  vector(const vector& __x);
};

class ClassName
{
private:
  vector m_namespace;
};

template<class T>
void make_pair(T y);

void add (const ClassName& pattern)
{
  make_pair (pattern);
}


///////////////////////////
// Testing another crash.

class CaloCell
{
public:
  CaloCell();
protected:
  union {
    int  m_quality ; 
  };
};


// Template class.
template <class T>
class C
{
public:
  C();
private:
  int x;
};
