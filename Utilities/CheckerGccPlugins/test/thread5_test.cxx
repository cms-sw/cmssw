// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// testing check_discarded_const_in_funcall

#pragma ATLAS check_thread_safety

const int* xx();
void f1(int* y);
void f2(const int* y)
{
  f1(const_cast<int*>(y));
}

void f3(const int y[10])
{
  f1((int*)&y[5]);
}


struct S
{
  int x;
  int foo();
};

void f4(const S* s)
{
  f1((int*)&s->x);
}


void f5(const int* y)
{
  int* yy [[gnu::thread_safe]] = const_cast<int*>(y);
  f1(yy);
}

void f6(const S& s)
{
  const_cast<S&>(s).foo();
}

void f7 [[gnu::argument_not_const_thread_safe]] (const int* y)
{
  f1(const_cast<int*>(y));
}

void f7a [[gnu::not_const_thread_safe]] (const int* y)
{
  f1(const_cast<int*>(y));
}

void f8 [[gnu::not_const_thread_safe]] ()
{
  const int* y = xx();
  f1(const_cast<int*>(y));
}

void f8a [[gnu::argument_not_const_thread_safe]] ()
{
  const int* y = xx();
  f1(const_cast<int*>(y));
}

void f9 [[gnu::not_thread_safe]] ()
{
  const int* y = xx();
  f1(const_cast<int*>(y));
}


struct str{ str(const char* s); };
void initParams (const str& name);

void foo()
{
  initParams("foo");
}


void retrieveAny(const int*& value);

void execute()
{ 
  const int* value=0;
  retrieveAny(value);
}

void foo (int const** p);

void bar()
{
  const int* pp = nullptr;
  foo (&pp);
}


struct F10
{
  void moveAux1 (int* const* beg);
  virtual void moveAux (int* const * beg)
  {
    moveAux1 (beg);
  }
};
F10 v10;



void operator delete(void*, unsigned long);
void foo (const int** it)
{
  delete *it;
}
void bar (const int* it)
{
  delete it;
}

struct M
{
  virtual void foo();
  virtual ~M();
  virtual void bar();
};
void bar (const M* it)
{
  delete it;
}


////////////////////////////////////////////////////////////////////////////////////
// Check for a false positive seen when a function return value is declared const.
//

struct new_allocator
{
  new_allocator();
  new_allocator(const new_allocator&);
};

struct basic_string
{
  new_allocator	_M_dataplus;
  void clear();
};


const basic_string fptest1()
{
  basic_string str;
  str.clear();
  return str;
}
