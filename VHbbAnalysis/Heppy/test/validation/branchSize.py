import ROOT, sys
f = ROOT.TFile(sys.argv[1])
t = f.Get("tree")

nev = t.GetEntries()

tot = 0
for br in t.GetListOfBranches():
    bytes = br.GetTotBytes()
    tot += bytes
    print "{0} {1:.2f} bytes/event".format(br.GetName(), bytes/float(nev))

print "total {0:.2f} bytes/event".format(tot/float(nev))
