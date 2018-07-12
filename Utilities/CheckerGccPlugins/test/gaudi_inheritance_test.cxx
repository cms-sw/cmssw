// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

class Algorithm {};
class Service {};
class AlgTool {};

class Foo1 : public Algorithm {};
class Foo2 : public Service {};
class Foo3 : public AlgTool {};

class AthAlgorithm : public Algorithm {};

template <typename BASE> class extends: public BASE {};
class A1 : public extends<Algorithm> {};
class A2 : public extends<AthAlgorithm> {};
