from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('geomDB',
                 True,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 """Boolean to decide which geometry and reco-material description to read:
                 True means the one coming from the DB via the GT, False means the one coming
                 from the files, either the central ones from the main release area, or the
                 local ones, if present. The choice of this paramenter will also alter the output
                 filename of the job.""")

