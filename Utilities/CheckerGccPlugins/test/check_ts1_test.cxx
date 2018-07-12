// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

#include "check_ts1_test1.h"
#include "Control/p1/p1/p1a.h"
#include "Control/p1/p1/p1b.h"
#include "Control/p2/p2/p2a.h"
#include "Control/p2/p2/p2b.h"

void f1 [[gnu::check_thread_safety_debug]] () {}
void f2 [[gnu::check_thread_safety_debug]] [[gnu::check_thread_safety]] () {}


struct [[gnu::check_thread_safety_debug]] C1
{
  void g1 [[gnu::check_thread_safety_debug]] () {}
  void g2 [[gnu::check_thread_safety_debug]] ();
  void g3();
  void g4 [[gnu::check_thread_safety]] [[gnu::check_thread_safety_debug]] ();
  void g5 [[gnu::not_thread_safe]] [[gnu::check_thread_safety_debug]] ();
};


void C1::g2() {}
void C1::g3 [[gnu::check_thread_safety_debug]] () {}
void C1::g4() {}


struct [[gnu::check_thread_safety]] C2
{
  void h1 [[gnu::check_thread_safety_debug]] () {}
  void h2();

  struct C3 {
    void h3();
  };

  void h4 [[gnu::not_thread_safe]] [[gnu::check_thread_safety_debug]] () {}
};

void C2::h2 [[gnu::check_thread_safety_debug]] () {}
void C2::C3::h3 [[gnu::check_thread_safety_debug]] () {}


