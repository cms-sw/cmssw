// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// testing check_discarded_const_from_return

#pragma ATLAS check_thread_safety

const int* xx();


int* f1()
{
  return const_cast<int*>(xx());
}


int* f2()
{
  int* y = const_cast<int*>(xx());
  return y;
}


int* f3 [[gnu::not_thread_safe]] ()
{
  int* y = const_cast<int*>(xx());
  return y;
}


int* f4 [[gnu::not_const_thread_safe]] ()
{
  int* y = const_cast<int*>(xx());
  return y;
}


int* f5 [[gnu::argument_not_const_thread_safe]] ()
{
  int* y = const_cast<int*>(xx());
  return y;
}

int* f6()
{
  int* y [[gnu::thread_safe]] = const_cast<int*>(xx());
  return y;
}


