#ifndef SimTracker_TrackHistory_CMSProcessTypes_h
#define SimTracker_TrackHistory_CMSProcessTypes_h

//! Struct holding legacy CMS convention for process types
struct CMS
{
    enum Process
    {
        Undefined = 0,
        Unknown,
        Primary,
        Hadronic,
        Decay,
        Compton,
        Annihilation,
        EIoni,
        HIoni,
        MuIoni,
        Photon,
        MuPairProd,
        Conversions,
        EBrem,
        SynchrotronRadiation,
        MuBrem,
        MuNucl
    };
};

#endif
