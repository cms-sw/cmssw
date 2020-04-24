
#ifndef TrackCategories_h
#define TrackCategories_h

#include <ostream>
#include <vector>

class TrackCategories
{

public:

    //! Categories available to vertex
    enum Category
    {
        Fake = 0,
        Reconstructed = Fake,
        Bad,
        BadInnerHits,
        SharedInnerHits,
        SignalEvent,
        Bottom,
        Charm,
        Light,
        Muon,
        TrackerSimHits,
        BWeakDecay,
        CWeakDecay,
        ChargePionDecay,
        ChargeKaonDecay,
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
        FromBWeakDecayMuon,
        FromCWeakDecayMuon,
        DecayOnFlightMuon,
        FromChargePionMuon,
        FromChargeKaonMuon,
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
    TrackCategories()
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
    void unknownTrack();

    //! Flag containers
    Flags flags_;

};

// Operation overload for printing the categories
std::ostream & operator<< (std::ostream &, TrackCategories const &);

#endif
