/*
 *  TrackCategories.h
 *
 *  Created by Victor Eduardo Bazterra on 5/29/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TrackCategories_h
#define TrackCategories_h

#include <iostream>
#include <string>
#include <vector>

namespace TrackCategories {

  enum Category {
    Fake = 0,
    Bad,
    SignalEvent,
    Bottom,
    Charm,
    Light,
    BWeakDecay,
    CWeakDecay,
    TauDecay,
    KsDecay,
    LambdaDecay,
    LongLivedDecay,
    Conversion,
    Interaction,
    PrimaryVertex,
    SecondaryVertex,
    TierciaryVertex,
    Unknown
  };

  typedef std::vector<bool> Flags;
}

std::ostream & operator<< (std::ostream & os, TrackCategories::Flags const &);

#endif

