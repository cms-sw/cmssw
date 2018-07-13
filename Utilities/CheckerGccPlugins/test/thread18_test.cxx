// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration.
// thread18_test: test call through function pointer.

#pragma ATLAS check_thread_safety

struct PyTypeObject;

typedef void (*newfunc)(PyTypeObject *);

struct PyTypeObject {
    newfunc tp_new;
};

extern PyTypeObject PyTuple_Type;


PyTypeObject TTupleOfInstances_Type;
void TTupleOfInstances_New(  )
{
  PyTuple_Type.tp_new( &TTupleOfInstances_Type);
}

