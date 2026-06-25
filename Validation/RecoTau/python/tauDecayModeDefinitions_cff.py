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


def makeDecayModePostProcessorProfiles():
    decayModeEfficiencyProfiles = []
    decayModeFakeProfiles = []
    decayModeSplitProfiles = []
    decayModeDuplicateProfiles = []
    decayModeResponseProfiles = []

    for dm in genDecayModes:
        for var, title in kinVars.items():
            decayModeEfficiencyProfiles.append(
                f"Eff_{dm}_vs_{var} 'Efficiency {dm} vs {title}' genTauMatched_{dm}_{var} genTau_{dm}_{var}"
            )

            decayModeSplitProfiles.append(
                f"Split_{dm}_vs_{var} 'Split Rate {dm} vs {title}' genTauMultiMatched_{dm}_{var} genTau_{dm}_{var}"
            )

            decayModeResponseProfiles.append(
                f"ResponsePt_{dm}_RecoOverGen_vs_{var} 'Response {dm} RecoOverGen vs {title}' responsePt_{dm}_{var} rms"
            )

            decayModeResponseProfiles.append(
                f"ResponseMass_{dm}_RecoOverGen_vs_{var} 'Mass response {dm} RecoOverGen vs {title}' responseMass_{dm}_{var} rms"
            )

    for dm in recoDecayModes:
        for var, title in kinVars.items():
            decayModeFakeProfiles.append(
                f"Fake_{dm}_vs_{var} 'Fake Rate {dm} vs {title}' recoTauMatched_{dm}_{var} recoTau_{dm}_{var} fake"
            )

            decayModeDuplicateProfiles.append(
                f"Dup_{dm}_vs_{var} 'Duplicate Rate {dm} vs {title}' recoTauMultiMatched_{dm}_{var} recoTau_{dm}_{var}"
            )

    return (
        decayModeEfficiencyProfiles,
        decayModeFakeProfiles,
        decayModeSplitProfiles,
        decayModeDuplicateProfiles,
        decayModeResponseProfiles,
    )