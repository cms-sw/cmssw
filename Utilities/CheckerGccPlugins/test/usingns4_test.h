// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

template <class T>
struct RegisterBaseInit
{
  RegisterBaseInit() __attribute__ ((used));
};


template <class T>
RegisterBaseInit<T>::RegisterBaseInit()
{
  BaseInfoBase::addInit(&typeid(int));
}
