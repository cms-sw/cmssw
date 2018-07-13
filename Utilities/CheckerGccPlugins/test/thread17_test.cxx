// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
// thread17_test: testing check_attrib_consistency

#pragma ATLAS check_thread_safety

void f1 [[gnu::not_const_thread_safe]] ();
void f1 [[gnu::not_reentrant]] () {}
