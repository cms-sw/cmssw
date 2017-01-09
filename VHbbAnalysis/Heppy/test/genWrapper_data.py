from vhbb import *

sample.isMC=False
sample.isData=True

from vhbb_combined import *
sample.json="json.txt"
TriggerObjectsAna.triggerObjectInputTag = ('selectedPatTrigger','','RECO')
FlagsAna.processName='RECO'
TrigAna.triggerBits = triggerTableData
from PhysicsTools.HeppyCore.framework.looper import Looper
looper = Looper('Loop', config, nPrint=0, nEvents=1)
of = open("tree_data.py", "w")
of.write(looper.analyzers[-1].getPythonWrapper())
of.close()
