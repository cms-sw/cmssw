// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

namespace rapidjson {

template <typename U> 
class GenericValue {
public:

~GenericValue();

template <typename T>
GenericValue& operator[](T*) {
  static GenericValue NullValue;
  return NullValue;
}

};


typedef GenericValue<int > Value;

}


