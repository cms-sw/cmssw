import ROOT
import sys

tf = ROOT.TFile(sys.argv[1])
tt = tf.Get("TH1Fs")
for ev in tt:
    print ev.FullName, ev.Value, ev.Value.GetMean()
tt = tf.Get("TH1Ds")
for ev in tt:
    print ev.FullName, ev.Value, ev.Value.GetMean()
