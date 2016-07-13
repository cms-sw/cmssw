from vhbb_combined import *
from PhysicsTools.HeppyCore.framework.looper import Looper
looper = Looper('Loop', config, nPrint=0, nEvents=1)
of = open("tree.py", "w")
of.write(looper.analyzers[-1].getPythonWrapper())
of.close()
