// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

template <class T>
const BaseInfoBase& BaseInfo<T>::baseinfo()
{
  BaseInfoBase* inst = s_instance.instance;
  return *inst;
}


template <class T>
typename BaseInfo<T>::instance_holder BaseInfo<T>::s_instance;
