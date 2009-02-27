
#ifndef VertexCategories_h
#define VertexCategories_h

#include <iostream>
#include <string>
#include <vector>

struct VertexCategories
{

    enum Category
    {
        Fake = 0,
        SignalEvent,
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
        TertiaryVertex,
        TierciaryVertex = TertiaryVertex,
        Unknown
    };

    static const char * Names[];

    typedef std::vector<bool> Flags;

};

#endif
