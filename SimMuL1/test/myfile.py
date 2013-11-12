from ROOT import *


file = TFile.Open("mytest_tuple.root")

dirAna = file.Get("GEMCSCTriggerEfficiencyTree")
if not dirAna:
    sys.exit('Directory %s does not exist.' %(dirAna))
    
treeHits = dirAna.Get("test")
if not treeHits:
    sys.exit('Tree %s does not exist.' %(treeHits))

#print treeHits.GetEntries()

for event in treeHits:
    print event.gem_sh_detUnitId.at(0).size()

    #at(0).at(4), event.gem_sh_detUnitId.at(0).at(5), event.gem_sh_detUnitId.at(1).at(4), event.gem_sh_detUnitId.at(1).at(5)
