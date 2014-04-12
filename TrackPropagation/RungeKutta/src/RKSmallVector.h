#ifndef RKSmallVector_H
#define RKSmallVector_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "DataFormats/GeometryVector/interface/PreciseFloatType.h"

#include <Math/SVector.h>


#include <iostream>

template<typename T, int N> using RKSmallVector =  ROOT::Math::SVector<T,N>;

#endif
