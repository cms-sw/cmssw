import FWCore.ParameterSet.Config as cms

NeutronSimHitAnalyzer = cms.EDAnalyzer("NeutronSimHitAnalyzer",
    inputIsNeutrons = cms.bool(True),
    defaultInputTagCSC = cms.InputTag("cscNeutronWriter",""),
    defaultInputTagRPC = cms.InputTag("rpcNeutronWriter",""),
    defaultInputTagGEM = cms.InputTag("gemNeutronWriter",""),
    defaultInputTagDT = cms.InputTag("dtNeutronWriter",""),
    inputTagCSC = cms.InputTag("g4SimHits","MuonCSCHits"),
    inputTagGEM = cms.InputTag("g4SimHits","MuonGEMHits"),
    inputTagRPC = cms.InputTag("g4SimHits","MuonRPCHits"),
    inputTagDT = cms.InputTag("g4SimHits","MuonDTHits"),
    ##
    maxCSCStations = cms.int32(4),
    nCSCTypes = cms.int32(10),
    maxDTStations = cms.int32(4),
    nDTTypes = cms.int32(4),
    maxRPCfStations = cms.int32(8),
    nRPCfTypes = cms.int32(8),
    maxRPCbStations = cms.int32(4),
    nRPCbTypes = cms.int32(12),
    maxGEMStations = cms.int32(1),
    nGEMTypes = cms.int32(1),
    ## chamber types
    cscTypesLong = cms.vstring("Any", "ME1/a", "ME1/b", "ME1/2", "ME1/3", "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1", "ME4/2"),
    cscTypesShort = cms.vstring("Any", "ME1a", "ME1b", "ME12", "ME13", "ME21", "ME22", "ME31", "ME32", "ME41", "ME42"),
    dtTypesLong = cms.vstring("Any", "MB1/0", "MB1/1", "MB1/2", "MB2/0", "MB2/1", "MB2/2", "MB3/0", "MB3/1", "MB3/2", "MB4/0", "MB4/1", "MB4/2"),
    dtTypesShort = cms.vstring("Any", "MB10", "MB11", "MB12", "MB20", "MB21", "MB22", "MB30", "MB31", "MB32", "MB40", "MB41", "MB42"),
    rpcfTypesLong = cms.vstring("Any", "RE1/2", "RE1/3", "RE2/2", "RE2/3", "RE3/2", "RE3/3", "RE4/2", "RE4/3"),
    rpcfTypesShort = cms.vstring("Any", "RE12", "RE13", "RE22", "RE23", "RE32", "RE33", "RE42", "RE43"),
    rpcbTypesLong = cms.vstring("Any", "RB1/0", "RB1/1", "RB1/2", "RB2/0", "RB2/1", "RB2/2", "RB3/0", "RB3/1", "RB3/2", "RB4/0", "RB4/1", "RB4/2"),
    rpcbTypesShort = cms.vstring("Any", "RB10", "RB11", "RB12", "RB20", "RB21", "RB22", "RB30", "RB31", "RB32", "RB40", "RB41", "RB42"),
    gemTypesLong = cms.vstring("Any", "GE1/1"),
    gemTypesShort = cms.vstring("Any", "GE11"),                                   
    ## add chamber areas
    pu = cms.int32(25),
    fractionEmptyBX = cms.double(0.77),
    bxRate = cms.int32(40000000),
    ## chamber areas
                                       ## are these sensitive volumes?
    cscAreascm2 = cms.vdouble(100000, 1068.32, 4108.77, 10872.75, 13559.025, 16986, 31121.2, 15716.4, 31121.2, 14542.5, 31121.2), ##why 100000 for ANY?
    rpfcAreascm2 = cms.vdouble(100000, 11700, 17360, 11690, 19660, 11690, 19660, 11690, 19660),
    gemAreascm2 = cms.vdouble(100000, 3226.446), ## current geometry
    ##gemAreascm2 = cms.vdouble(100000, 4082.091), ## extended geometry    cscRadialSegmentation = cms.vint32(100, 36, 36, 36, 36, 18, 36, 18, 36, 18, 36), ##why 100? for ANY?
    rpcfRadialSegmentation = cms.int32(100, 36),
    gemRadialSegmentation = cms.int32(100, 36),
    ##layers
    nRPCLayers = cms.int32(1),
    nGEMLayers = cms.int32(2),
    nCSCLayers = cms.int32(6)                                   
)

