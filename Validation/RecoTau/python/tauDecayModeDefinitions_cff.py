genDecayModes = [
    "oneProng0Pi0",
    "oneProng1Pi0",
    "oneProng2Pi0",
    "oneProngOther",
    "threeProng0Pi0",
    "threeProng1Pi0",
    "threeProngOther",
    "rare",
]

recoDecayModes = [
    "oneProng0Pi0",
    "oneProng1Pi0",
    "oneProng2Pi0",
    "oneProngOther",
    "threeProng0Pi0",
    "threeProng1Pi0",
    "threeProngOther",
    "rare",
    "unknown",
]

decayModes = recoDecayModes

kinVars = {
    "pt": "p_{T}",
    "eta": "#eta",
    "phi": "#phi",
    "mass": "mass",
}


def makeDecayModePostProcessorProfiles(subDirectory="DecayModes"):
    # Paths are relative to each DQMGenericClient subDir. Prefix both the
    # source histograms and the harvested outputs, including response profiles.
    prefix = subDirectory.strip("/")
    prefix = f"{prefix}/" if prefix else ""
    decayModeEfficiencyProfiles = []
    decayModeFakeProfiles = []
    decayModeSplitProfiles = []
    decayModeDuplicateProfiles = []
    decayModeResponseProfiles = []

    for dm in genDecayModes:
        for var, title in kinVars.items():
            decayModeEfficiencyProfiles.append(
                f"{prefix}Eff_{dm}_vs_{var} 'Efficiency {dm} vs {title}' {prefix}genTauMatched_{dm}_{var} {prefix}genTau_{dm}_{var}"
            )

            decayModeSplitProfiles.append(
                f"{prefix}Split_{dm}_vs_{var} 'Split Rate {dm} vs {title}' {prefix}genTauMultiMatched_{dm}_{var} {prefix}genTau_{dm}_{var}"
            )

            decayModeResponseProfiles.append(
                f"{prefix}ResponsePt_{dm}_RecoOverGen_vs_{var} 'Response {dm} RecoOverGen vs {title}' {prefix}responsePt_{dm}_{var} rms"
            )

            decayModeResponseProfiles.append(
                f"{prefix}ResponseMass_{dm}_RecoOverGen_vs_{var} 'Mass response {dm} RecoOverGen vs {title}' {prefix}responseMass_{dm}_{var} rms"
            )

    for dm in recoDecayModes:
        for var, title in kinVars.items():
            decayModeFakeProfiles.append(
                f"{prefix}Fake_{dm}_vs_{var} 'Fake Rate {dm} vs {title}' {prefix}recoTauMatched_{dm}_{var} {prefix}recoTau_{dm}_{var} fake"
            )

            decayModeDuplicateProfiles.append(
                f"{prefix}Dup_{dm}_vs_{var} 'Duplicate Rate {dm} vs {title}' {prefix}recoTauMultiMatched_{dm}_{var} {prefix}recoTau_{dm}_{var}"
            )

    return (
        decayModeEfficiencyProfiles,
        decayModeFakeProfiles,
        decayModeSplitProfiles,
        decayModeDuplicateProfiles,
        decayModeResponseProfiles,
    )
