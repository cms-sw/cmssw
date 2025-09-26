import FWCore.ParameterSet.Config as cms

HLTGenResSource = cms.EDProducer("HLTGenResSource",
    hltProcessName = cms.string('HLT'),
    resCollConfigs = cms.VPSet(
        cms.PSet(
            collectionName = cms.string('hltEgammaCandidatesL1Seeded'),
            histConfigs = cms.VPSet(cms.PSet(
                rangeCuts = cms.VPSet(cms.PSet(
                    allowedRanges = cms.vstring(
                        '-1.4442:1.4442',
                        '1.566:2.5',
                        '-2.5:-1.566'
                    ),
                    rangeVar = cms.string('eta')
                )),
                resBinLowEdges = cms.vdouble(
                    0.0, 0.01, 0.02, 0.03, 0.04,
                    0.05, 0.06, 0.07, 0.08, 0.09,
                    0.1, 0.11, 0.12, 0.13, 0.14,
                    0.15, 0.16, 0.17, 0.18, 0.19,
                    0.2, 0.21, 0.22, 0.23, 0.24,
                    0.25, 0.26, 0.27, 0.28, 0.29,
                    0.3, 0.31, 0.32, 0.33, 0.34,
                    0.35000000000000003, 0.36, 0.37, 0.38, 0.39,
                    0.4, 0.41000000000000003, 0.42, 0.43, 0.44,
                    0.45, 0.46, 0.47000000000000003, 0.48, 0.49,
                    0.5, 0.51, 0.52, 0.53, 0.54,
                    0.55, 0.56, 0.5700000000000001, 0.58, 0.59,
                    0.6, 0.61, 0.62, 0.63, 0.64,
                    0.65, 0.66, 0.67, 0.68, 0.6900000000000001,
                    0.7000000000000001, 0.71, 0.72, 0.73, 0.74,
                    0.75, 0.76, 0.77, 0.78, 0.79,
                    0.8, 0.81, 0.8200000000000001, 0.8300000000000001, 0.84,
                    0.85, 0.86, 0.87, 0.88, 0.89,
                    0.9, 0.91, 0.92, 0.93, 0.9400000000000001,
                    0.9500000000000001, 0.96, 0.97, 0.98, 0.99,
                    1.0, 1.01, 1.02, 1.03, 1.04,
                    1.05, 1.06, 1.07, 1.08, 1.09,
                    1.1, 1.11, 1.12, 1.1300000000000001, 1.1400000000000001,
                    1.1500000000000001, 1.16, 1.17, 1.18, 1.19,
                    1.2, 1.21, 1.22, 1.23, 1.24,
                    1.25, 1.26, 1.27, 1.28, 1.29,
                    1.3, 1.31, 1.32, 1.33, 1.34,
                    1.35, 1.36, 1.37, 1.3800000000000001, 1.3900000000000001,
                    1.4000000000000001, 1.41, 1.42, 1.43, 1.44,
                    1.45, 1.46, 1.47, 1.48, 1.49,
                    1.5
                ),
                resVar = cms.string('ptRes'),
                vsBinLowEdges = cms.vdouble(
                    0, 5, 10, 15, 20,
                    25, 30, 40, 50, 75,
                    100, 150, 400, 800, 1500,
                    10000
                ),
                vsVar = cms.string('pt')
            )),
            objType = cms.string('ele')
        ),
        cms.PSet(
            collectionName = cms.string('hltEgammaCandidatesL1Seeded'),
            histConfigs = cms.VPSet(cms.PSet(
                rangeCuts = cms.VPSet(cms.PSet(
                    allowedRanges = cms.vstring(
                        '-1.4442:1.4442',
                        '1.566:2.5',
                        '-2.5:-1.566'
                    ),
                    rangeVar = cms.string('eta')
                )),
                resBinLowEdges = cms.vdouble(
                    0.0, 0.01, 0.02, 0.03, 0.04,
                    0.05, 0.06, 0.07, 0.08, 0.09,
                    0.1, 0.11, 0.12, 0.13, 0.14,
                    0.15, 0.16, 0.17, 0.18, 0.19,
                    0.2, 0.21, 0.22, 0.23, 0.24,
                    0.25, 0.26, 0.27, 0.28, 0.29,
                    0.3, 0.31, 0.32, 0.33, 0.34,
                    0.35000000000000003, 0.36, 0.37, 0.38, 0.39,
                    0.4, 0.41000000000000003, 0.42, 0.43, 0.44,
                    0.45, 0.46, 0.47000000000000003, 0.48, 0.49,
                    0.5, 0.51, 0.52, 0.53, 0.54,
                    0.55, 0.56, 0.5700000000000001, 0.58, 0.59,
                    0.6, 0.61, 0.62, 0.63, 0.64,
                    0.65, 0.66, 0.67, 0.68, 0.6900000000000001,
                    0.7000000000000001, 0.71, 0.72, 0.73, 0.74,
                    0.75, 0.76, 0.77, 0.78, 0.79,
                    0.8, 0.81, 0.8200000000000001, 0.8300000000000001, 0.84,
                    0.85, 0.86, 0.87, 0.88, 0.89,
                    0.9, 0.91, 0.92, 0.93, 0.9400000000000001,
                    0.9500000000000001, 0.96, 0.97, 0.98, 0.99,
                    1.0, 1.01, 1.02, 1.03, 1.04,
                    1.05, 1.06, 1.07, 1.08, 1.09,
                    1.1, 1.11, 1.12, 1.1300000000000001, 1.1400000000000001,
                    1.1500000000000001, 1.16, 1.17, 1.18, 1.19,
                    1.2, 1.21, 1.22, 1.23, 1.24,
                    1.25, 1.26, 1.27, 1.28, 1.29,
                    1.3, 1.31, 1.32, 1.33, 1.34,
                    1.35, 1.36, 1.37, 1.3800000000000001, 1.3900000000000001,
                    1.4000000000000001, 1.41, 1.42, 1.43, 1.44,
                    1.45, 1.46, 1.47, 1.48, 1.49,
                    1.5
                ),
                resVar = cms.string('ptRes'),
                vsBinLowEdges = cms.vdouble(
                    0, 5, 10, 15, 20,
                    25, 30, 40, 50, 75,
                    100, 150, 400, 800, 1500,
                    10000
                ),
                vsVar = cms.string('pt')
            )),
            objType = cms.string('pho')
        ),
        cms.PSet(
            collectionName = cms.string('hltPhase2L3MuonCandidates'),
            histConfigs = cms.VPSet(cms.PSet(
                rangeCuts = cms.VPSet(cms.PSet(
                    allowedRanges = cms.vstring('-2.4:2.4'),
                    rangeVar = cms.string('eta')
                )),
                resBinLowEdges = cms.vdouble(
                    0.0, 0.01, 0.02, 0.03, 0.04,
                    0.05, 0.06, 0.07, 0.08, 0.09,
                    0.1, 0.11, 0.12, 0.13, 0.14,
                    0.15, 0.16, 0.17, 0.18, 0.19,
                    0.2, 0.21, 0.22, 0.23, 0.24,
                    0.25, 0.26, 0.27, 0.28, 0.29,
                    0.3, 0.31, 0.32, 0.33, 0.34,
                    0.35000000000000003, 0.36, 0.37, 0.38, 0.39,
                    0.4, 0.41000000000000003, 0.42, 0.43, 0.44,
                    0.45, 0.46, 0.47000000000000003, 0.48, 0.49,
                    0.5, 0.51, 0.52, 0.53, 0.54,
                    0.55, 0.56, 0.5700000000000001, 0.58, 0.59,
                    0.6, 0.61, 0.62, 0.63, 0.64,
                    0.65, 0.66, 0.67, 0.68, 0.6900000000000001,
                    0.7000000000000001, 0.71, 0.72, 0.73, 0.74,
                    0.75, 0.76, 0.77, 0.78, 0.79,
                    0.8, 0.81, 0.8200000000000001, 0.8300000000000001, 0.84,
                    0.85, 0.86, 0.87, 0.88, 0.89,
                    0.9, 0.91, 0.92, 0.93, 0.9400000000000001,
                    0.9500000000000001, 0.96, 0.97, 0.98, 0.99,
                    1.0, 1.01, 1.02, 1.03, 1.04,
                    1.05, 1.06, 1.07, 1.08, 1.09,
                    1.1, 1.11, 1.12, 1.1300000000000001, 1.1400000000000001,
                    1.1500000000000001, 1.16, 1.17, 1.18, 1.19,
                    1.2, 1.21, 1.22, 1.23, 1.24,
                    1.25, 1.26, 1.27, 1.28, 1.29,
                    1.3, 1.31, 1.32, 1.33, 1.34,
                    1.35, 1.36, 1.37, 1.3800000000000001, 1.3900000000000001,
                    1.4000000000000001, 1.41, 1.42, 1.43, 1.44,
                    1.45, 1.46, 1.47, 1.48, 1.49,
                    1.5
                ),
                resVar = cms.string('ptRes'),
                vsBinLowEdges = cms.vdouble(
                    0, 5, 10, 15, 20,
                    25, 30, 40, 50, 75,
                    100, 150, 400, 800, 1500,
                    10000
                ),
                vsVar = cms.string('pt')
            )),
            objType = cms.string('mu')
        ),
        cms.PSet(
            collectionName = cms.string('hltHpsPFTauProducer'),
            histConfigs = cms.VPSet(cms.PSet(
                rangeCuts = cms.VPSet(cms.PSet(
                    allowedRanges = cms.vstring('-2.4:2.4'),
                    rangeVar = cms.string('eta')
                )),
                resBinLowEdges = cms.vdouble(
                    0.0, 0.01, 0.02, 0.03, 0.04,
                    0.05, 0.06, 0.07, 0.08, 0.09,
                    0.1, 0.11, 0.12, 0.13, 0.14,
                    0.15, 0.16, 0.17, 0.18, 0.19,
                    0.2, 0.21, 0.22, 0.23, 0.24,
                    0.25, 0.26, 0.27, 0.28, 0.29,
                    0.3, 0.31, 0.32, 0.33, 0.34,
                    0.35000000000000003, 0.36, 0.37, 0.38, 0.39,
                    0.4, 0.41000000000000003, 0.42, 0.43, 0.44,
                    0.45, 0.46, 0.47000000000000003, 0.48, 0.49,
                    0.5, 0.51, 0.52, 0.53, 0.54,
                    0.55, 0.56, 0.5700000000000001, 0.58, 0.59,
                    0.6, 0.61, 0.62, 0.63, 0.64,
                    0.65, 0.66, 0.67, 0.68, 0.6900000000000001,
                    0.7000000000000001, 0.71, 0.72, 0.73, 0.74,
                    0.75, 0.76, 0.77, 0.78, 0.79,
                    0.8, 0.81, 0.8200000000000001, 0.8300000000000001, 0.84,
                    0.85, 0.86, 0.87, 0.88, 0.89,
                    0.9, 0.91, 0.92, 0.93, 0.9400000000000001,
                    0.9500000000000001, 0.96, 0.97, 0.98, 0.99,
                    1.0, 1.01, 1.02, 1.03, 1.04,
                    1.05, 1.06, 1.07, 1.08, 1.09,
                    1.1, 1.11, 1.12, 1.1300000000000001, 1.1400000000000001,
                    1.1500000000000001, 1.16, 1.17, 1.18, 1.19,
                    1.2, 1.21, 1.22, 1.23, 1.24,
                    1.25, 1.26, 1.27, 1.28, 1.29,
                    1.3, 1.31, 1.32, 1.33, 1.34,
                    1.35, 1.36, 1.37, 1.3800000000000001, 1.3900000000000001,
                    1.4000000000000001, 1.41, 1.42, 1.43, 1.44,
                    1.45, 1.46, 1.47, 1.48, 1.49,
                    1.5
                ),
                resVar = cms.string('ptRes'),
                vsBinLowEdges = cms.vdouble(
                    0, 5, 10, 15, 20,
                    25, 30, 40, 50, 75,
                    100, 150, 400, 800, 1500,
                    10000
                ),
                vsVar = cms.string('pt')
            )),
            objType = cms.string('tau')
        ),
        cms.PSet(
            collectionName = cms.string('hltAK4PFPuppiJetsCorrected'),
            histConfigs = cms.VPSet(cms.PSet(
                rangeCuts = cms.VPSet(cms.PSet(
                    allowedRanges = cms.vstring('-2.4:2.4'),
                    rangeVar = cms.string('eta')
                )),
                resBinLowEdges = cms.vdouble(
                    0.0, 0.01, 0.02, 0.03, 0.04,
                    0.05, 0.06, 0.07, 0.08, 0.09,
                    0.1, 0.11, 0.12, 0.13, 0.14,
                    0.15, 0.16, 0.17, 0.18, 0.19,
                    0.2, 0.21, 0.22, 0.23, 0.24,
                    0.25, 0.26, 0.27, 0.28, 0.29,
                    0.3, 0.31, 0.32, 0.33, 0.34,
                    0.35000000000000003, 0.36, 0.37, 0.38, 0.39,
                    0.4, 0.41000000000000003, 0.42, 0.43, 0.44,
                    0.45, 0.46, 0.47000000000000003, 0.48, 0.49,
                    0.5, 0.51, 0.52, 0.53, 0.54,
                    0.55, 0.56, 0.5700000000000001, 0.58, 0.59,
                    0.6, 0.61, 0.62, 0.63, 0.64,
                    0.65, 0.66, 0.67, 0.68, 0.6900000000000001,
                    0.7000000000000001, 0.71, 0.72, 0.73, 0.74,
                    0.75, 0.76, 0.77, 0.78, 0.79,
                    0.8, 0.81, 0.8200000000000001, 0.8300000000000001, 0.84,
                    0.85, 0.86, 0.87, 0.88, 0.89,
                    0.9, 0.91, 0.92, 0.93, 0.9400000000000001,
                    0.9500000000000001, 0.96, 0.97, 0.98, 0.99,
                    1.0, 1.01, 1.02, 1.03, 1.04,
                    1.05, 1.06, 1.07, 1.08, 1.09,
                    1.1, 1.11, 1.12, 1.1300000000000001, 1.1400000000000001,
                    1.1500000000000001, 1.16, 1.17, 1.18, 1.19,
                    1.2, 1.21, 1.22, 1.23, 1.24,
                    1.25, 1.26, 1.27, 1.28, 1.29,
                    1.3, 1.31, 1.32, 1.33, 1.34,
                    1.35, 1.36, 1.37, 1.3800000000000001, 1.3900000000000001,
                    1.4000000000000001, 1.41, 1.42, 1.43, 1.44,
                    1.45, 1.46, 1.47, 1.48, 1.49,
                    1.5
                ),
                resVar = cms.string('ptRes'),
                vsBinLowEdges = cms.vdouble(
                    0, 5, 10, 15, 20,
                    25, 30, 40, 50, 75,
                    100, 150, 400, 800, 1500,
                    10000
                ),
                vsVar = cms.string('pt')
            )),
            objType = cms.string('AK4jet')
        )
    )
)


HLTGenValSourceAK4 = cms.EDProducer("HLTGenValSource",
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(
        cms.PSet(
            binLowEdges = cms.vdouble(
                0, 100, 200, 300, 350,
                375, 400, 425, 450, 475,
                500, 550, 600, 700, 800,
                900, 1000
            ),
            rangeCuts = cms.VPSet(cms.PSet(
                allowedRanges = cms.vstring('-5:5'),
                rangeVar = cms.string('eta')
            )),
            vsVar = cms.string('pt')
        ),
        cms.PSet(
            binLowEdges = cms.vdouble(
                -4, -3, -2.5, -2, -1.5,
                -1, -0.5, 0, 0.5, 1,
                1.5, 2, 2.5, 3, 4
            ),
            rangeCuts = cms.VPSet(cms.PSet(
                allowedRanges = cms.vstring('200:9999'),
                rangeVar = cms.string('pt')
            )),
            vsVar = cms.string('eta')
        )
    ),
    hltPathsToCheck = cms.vstring('HLT_AK4PFPuppiJet520'),
    hltProcessName = cms.string('HLT'),
    objType = cms.string('AK4jet'),
    sampleLabel = cms.string(''),
    trigEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT")
)


HLTGenValSourceAK8 = cms.EDProducer("HLTGenValSource",
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(
        cms.PSet(
            binLowEdges = cms.vdouble(
                0, 100, 200, 300, 350,
                375, 400, 425, 450, 475,
                500, 550, 600, 700, 800,
                900, 1000
            ),
            rangeCuts = cms.VPSet(cms.PSet(
                allowedRanges = cms.vstring('-5:5'),
                rangeVar = cms.string('eta')
            )),
            vsVar = cms.string('pt')
        ),
        cms.PSet(
            binLowEdges = cms.vdouble(
                -4, -3, -2.5, -2, -1.5,
                -1, -0.5, 0, 0.5, 1,
                1.5, 2, 2.5, 3, 4
            ),
            rangeCuts = cms.VPSet(cms.PSet(
                allowedRanges = cms.vstring('200:9999'),
                rangeVar = cms.string('pt')
            )),
            vsVar = cms.string('eta')
        )
    ),
    hltPathsToCheck = cms.vstring(
        'HLT_AK8PFJet500',
        'HLT_AK8PFJet400_TrimMass30:minMass=50'
    ),
    hltProcessName = cms.string('HLT'),
    objType = cms.string('AK8jet'),
    sampleLabel = cms.string(''),
    trigEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT")
)


HLTGenValSourceELE = cms.EDProducer("HLTGenValSource",
    binnings = cms.VPSet(cms.PSet(
        binLowEdges = cms.vdouble(
            0, 10, 20, 25, 30,
            35, 40, 45, 50, 55,
            60, 65, 70, 75, 80,
            85, 90, 95, 100, 105,
            110, 115, 120, 125, 130,
            135, 140, 145, 150
        ),
        name = cms.string('ptBins'),
        vsVar = cms.string('pt')
    )),
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(
        cms.PSet(
            binLowEdges = cms.vdouble(
                0, 100, 200, 300, 400,
                500, 600, 700, 800, 900,
                1000, 1100, 1200, 1300, 1400,
                1500, 1600, 1700, 1800, 1900,
                2000, 2100, 2200, 2300, 2400,
                2500, 2600, 2700, 2800, 2900,
                3000, 3100, 3200, 3300, 3400,
                3500, 3600, 3700, 3800, 3900
            ),
            rangeCuts = cms.VPSet(cms.PSet(
                allowedRanges = cms.vstring(
                    '-1.4442:1.4442',
                    '1.566:2.5',
                    '-2.5:-1.566'
                ),
                rangeVar = cms.string('eta')
            )),
            vsVar = cms.string('pt')
        ),
        cms.PSet(
            binLowEdges = cms.vdouble(
                -4, -3, -2.5, -2, -1.5,
                -1, -0.5, 0, 0.5, 1,
                1.5, 2, 2.5, 3, 4
            ),
            rangeCuts = cms.VPSet(cms.PSet(
                allowedRanges = cms.vstring('40:9999'),
                rangeVar = cms.string('pt')
            )),
            vsVar = cms.string('eta')
        )
    ),
    histConfigs2D = cms.VPSet(cms.PSet(
        binLowEdgesX = cms.vdouble(
            -4, -3, -2.5, -2, -1.5,
            -1, -0.5, 0, 0.5, 1,
            1.5, 2, 2.5, 3, 4
        ),
        binLowEdgesY = cms.vdouble(
            -3.2, -3.0, -2.8000000000000003, -2.6, -2.4000000000000004,
            -2.2, -2.0, -1.8, -1.6, -1.4000000000000001,
            -1.2000000000000002, -1.0, -0.7999999999999998, -0.6000000000000001, -0.3999999999999999,
            -0.20000000000000018, 0.0, 0.20000000000000018, 0.3999999999999999, 0.6000000000000001,
            0.7999999999999998, 1.0, 1.2000000000000002, 1.4000000000000004, 1.6000000000000005,
            1.7999999999999998, 2.0, 2.2, 2.4000000000000004, 2.6000000000000005,
            2.8, 3.0, 3.2
        ),
        vsVarX = cms.string('eta'),
        vsVarY = cms.string('phi')
    )),
    hltPathsToCheck = cms.vstring(
        'HLT_Ele115_NonIso_L1Seeded',
        'HLT_Ele26_WP70_L1Seeded',
        'HLT_Ele26_WP70_L1Seeded:region=EB,tag=barrel',
        'HLT_Ele26_WP70_L1Seeded:region=EE,tag=endcap',
        'HLT_Ele26_WP70_L1Seeded:bins=ptBins,region=EB,tag=lowpt_barrel',
        'HLT_Ele26_WP70_L1Seeded:bins=ptBins,region=EE,tag=lowpt_endcap',
        'HLT_Ele32_WPTight_L1Seeded',
        'HLT_Ele32_WPTight_L1Seeded:region=EB,tag=barrel',
        'HLT_Ele32_WPTight_L1Seeded:region=EE,tag=endcap',
        'HLT_Ele32_WPTight_L1Seeded:bins=ptBins,region=EB,tag=lowpt_barrel',
        'HLT_Ele32_WPTight_L1Seeded:bins=ptBins,region=EE,tag=lowpt_endcap',
        'HLT_Photon108EB_TightID_TightIso_L1Seeded:region=EB,ptMin=120,tag=barrel',
        'HLT_Photon187_L1Seeded:region=EB,ptMin=200,tag=barrel',
        'HLT_Photon187_L1Seeded:region=EE,ptMin=200,tag=endcap'
    ),
    hltProcessName = cms.string('HLT'),
    objType = cms.string('ele'),
    sampleLabel = cms.string(''),
    trigEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT")
)


HLTGenValSourceHT = cms.EDProducer("HLTGenValSource",
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(cms.PSet(
        binLowEdges = cms.vdouble(
            0, 100, 200, 300, 400,
            500, 600, 700, 800, 900,
            950, 1000, 1050, 1100, 1150,
            1200, 1300
        ),
        vsVar = cms.string('pt')
    )),
    hltPathsToCheck = cms.vstring('HLT_PFPuppiHT1070'),
    hltProcessName = cms.string('HLT'),
    objType = cms.string('AK4HT'),
    sampleLabel = cms.string(''),
    trigEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT")
)


HLTGenValSourceMET = cms.EDProducer("HLTGenValSource",
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(cms.PSet(
        binLowEdges = cms.vdouble(
            100, 120, 140, 160, 180,
            200, 220, 240, 260, 280,
            300, 320, 340, 360, 380,
            400, 420, 440, 460, 480
        ),
        vsVar = cms.string('pt')
    )),
    hltPathsToCheck = cms.vstring('HLT_PFPuppiMETTypeOne140_PFPuppiMHT140'),
    hltProcessName = cms.string('HLT'),
    objType = cms.string('MET'),
    sampleLabel = cms.string(''),
    trigEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT")
)


HLTGenValSourceMU = cms.EDProducer("HLTGenValSource",
    binnings = cms.VPSet(cms.PSet(
        binLowEdges = cms.vdouble(
            0, 100, 200, 300, 400,
            500, 600, 700, 800, 900,
            1000, 1100, 1200, 1300, 1400,
            1500, 1600, 1700, 1800, 1900,
            2000, 2100, 2200, 2300, 2400,
            2500, 2600, 2700, 2800, 2900,
            3000, 3100, 3200, 3300, 3400,
            3500, 3600, 3700, 3800, 3900
        ),
        name = cms.string('ptBinsHighPt'),
        vsVar = cms.string('pt')
    )),
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(
        cms.PSet(
            binLowEdges = cms.vdouble(
                0, 10, 20, 25, 30,
                35, 40, 45, 50, 55,
                60, 65, 70, 75, 80,
                85, 90, 95, 100, 105,
                110, 115, 120, 125, 130,
                135, 140, 145, 150
            ),
            rangeCuts = cms.VPSet(cms.PSet(
                allowedRanges = cms.vstring('-2.4:2.4'),
                rangeVar = cms.string('eta')
            )),
            vsVar = cms.string('pt')
        ),
        cms.PSet(
            binLowEdges = cms.vdouble(
                -4, -3, -2.5, -2, -1.5,
                -1, -0.5, 0, 0.5, 1,
                1.5, 2, 2.5, 3, 4
            ),
            vsVar = cms.string('eta')
        )
    ),
    hltPathsToCheck = cms.vstring(
        'HLT_IsoMu24_FromL1TkMuon:ptMin=30',
        'HLT_Mu50_FromL1TkMuon:ptMin=60,absEtaCut=1.2,tag=centralbarrel',
        'HLT_Mu50_FromL1TkMuon:ptMin=60,bins=ptBinsHighPt,tag=highpt_bins'
    ),
    hltProcessName = cms.string('HLT'),
    objType = cms.string('mu'),
    sampleLabel = cms.string(''),
    trigEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT")
)


HLTGenValSourceTAU = cms.EDProducer("HLTGenValSource",
    doOnlyLastFilter = cms.bool(False),
    histConfigs = cms.VPSet(
        cms.PSet(
            binLowEdges = cms.vdouble(
                0, 10, 20, 25, 30,
                35, 40, 45, 50, 55,
                60, 65, 70, 75, 80,
                85, 90, 95, 100, 105,
                110, 115, 120, 125, 130,
                135, 140, 145, 150
            ),
            rangeCuts = cms.VPSet(cms.PSet(
                allowedRanges = cms.vstring('-2.1:2.1'),
                rangeVar = cms.string('eta')
            )),
            vsVar = cms.string('pt')
        ),
        cms.PSet(
            binLowEdges = cms.vdouble(
                -4, -3, -2.5, -2, -1.5,
                -1, -0.5, 0, 0.5, 1,
                1.5, 2, 2.5, 3, 4
            ),
            rangeCuts = cms.VPSet(cms.PSet(
                allowedRanges = cms.vstring('50:9999'),
                rangeVar = cms.string('pt')
            )),
            vsVar = cms.string('eta')
        )
    ),
    hltPathsToCheck = cms.vstring(
        'HLT_DoubleMediumChargedIsoPFTauHPS40_eta2p1',
        'HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1'
    ),
    hltProcessName = cms.string('HLT'),
    objType = cms.string('tauHAD'),
    sampleLabel = cms.string(''),
    trigEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT")
)

from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting

# List of module names (as strings)
hltGenValSourceLabels = [
    'HLTGenValSourceMU',
    'HLTGenValSourceELE',
    'HLTGenValSourceTAU',
    'HLTGenValSourceHT',
    'HLTGenValSourceAK4',
    'HLTGenValSourceAK8',
    'HLTGenValSourceMET'
]

# change the path to monitor in the case of NGT scouting
for label in hltGenValSourceLabels:
    if label in globals():
        ngtScouting.toModify(globals()[label],
                             hltPathsToCheck = ['DST_PFScouting'])

from RecoMET.Configuration.RecoGenMET_cff import genMetCalo,genMetTrue
from RecoMET.Configuration.GenMETParticles_cff import genCandidatesForMET, genParticlesForMETAllVisible
from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets

hltGENValidation = cms.Sequence(genCandidatesForMET+
                                genParticlesForMETAllVisible+
                                genMetCalo+
                                genMetTrue+
                                tauGenJets+
                                HLTGenResSource+
                                HLTGenValSourceMU+
                                HLTGenValSourceELE+
                                HLTGenValSourceTAU+
                                HLTGenValSourceHT+
                                HLTGenValSourceAK4+
                                #HLTGenValSourceAK8  # uncomment if needed
                                HLTGenValSourceMET)
