
#ifndef VertexCategories_h
#define VertexCategories_h

#include <vector>
#include <ostream>

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
        JpsiDecay,
        XiDecay,
        OmegaDecay,
        SigmaPlusDecay,
        SigmaMinusDecay,
        LongLivedDecay,
        KnownProcess,
        UndefinedProcess,
        UnknownProcess,
        PrimaryProcess,
        HadronicProcess,
        DecayProcess,
        ComptonProcess,
        AnnihilationProcess,
        EIoniProcess,
        HIoniProcess,
        MuIoniProcess,
        PhotonProcess,
        MuPairProdProcess,
        ConversionsProcess,
        EBremProcess,
        SynchrotronRadiationProcess,
        MuBremProcess,
        MuNuclProcess,
        PrimaryVertex,
        SecondaryVertex,
        TertiaryVertex,
        TierciaryVertex = TertiaryVertex,
        Unknown
    };

    //! Name of the different categories
    static const char * const Names[];

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
