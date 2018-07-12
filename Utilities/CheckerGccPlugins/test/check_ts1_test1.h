// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

#pragma ATLAS check_thread_safety

void i1 [[gnu::check_thread_safety_debug]] () {}

struct [[gnu::not_thread_safe]] [[gnu::check_thread_safety_debug]] C3
{
  void j1 [[gnu::check_thread_safety_debug]] () {}
};

