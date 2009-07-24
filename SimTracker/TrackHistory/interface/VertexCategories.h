
#ifndef VertexCategories_h
#define VertexCategories_h

#include <vector>

class VertexCategories
{

public:

    //! Categories available to vertexes
    enum Category
    {
        Fake = 0,
        Reconstructed = Fake,
        SignalEvent,
        BWeakDecay,
        CWeakDecay,
        TauDecay,
        KsDecay,
        LambdaDecay,
        Jpsi,
        Xi,
        Omega,
        SigmaPlus,
        SigmaMinus,
        LongLivedDecay,
        Conversion,
        Interaction,
        PrimaryVertex,
        SecondaryVertex,
        TertiaryVertex,
        TierciaryVertex = TertiaryVertex,
        Unknown
    };

    //! Name of the different categories
    static const char * Names[];

    //! Main types associated to the class
    typedef std::vector<bool> Flags;

    //! Void constructor
    VertexCategories()
    {
        reset();
    }

    //! Returns track flag for a given category
    bool is(Category category) const
    {
        return flags_[category];
    }

    //! Returns flags with the category descriptions
    const Flags & flags() const
    {
        return flags_;
    }

protected:

    //! Reset the categories flags
    void reset()
    {
        flags_ = Flags(Unknown + 1, false);
    }

    // Check for unkown classification
    void unknownVertex();

    //! Flag containers
    Flags flags_;

};

// Operation overload for printing the categories
std::ostream & operator<< (std::ostream &, VertexCategories const &);

#endif
