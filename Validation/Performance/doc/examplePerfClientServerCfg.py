# This config file is used to define a set of perfsuite jobs
#    These may be run on a single machine or multiple machines depending on the inputs to cmsPerfClient.py
# The set of jobs is defined as a variable called listperfsuitekeywords
#
# listperfsuitekeywords items must all be dictionaries. Each dictionary can only use the following valid keywords:
# "castordir" (Path String), "TimeSizeEvents" (int), "IgProfEvents" (int), "ValgrindEvents" (int),
# "cmsScimark" (int), "cmsScimarkLarge" (int), "cmsdriverOptions" (string), "stepOptions" (string), "quicktest" (boolen),
# "profilers" (string of ints), "cpus" list of (int)s, "cores" (int), "prevrel" (path string), "isAllCandles" (boolean),
# "candles" list of (string)s, "bypasshlt" (boolean), "runonspare" (boolean)
#
#  Consult cmsPerfSuite.py --help for an explanation of these options
#
#  For example a default perfsuite run followed by a default run with only 50 TimeSize events would be
# 
# listperfsuitekeywords = [{},                     # An empty dictionary means run the default values
#                          {"TimeSizeEvents" : 50}]
#
#
#  A set of commands that:
#          1) Runs a default perfsuite run
#          2) Runs perfsuite with 25 TimeSize Events, on cores 1 & 2, for only MinBias candle and for the GEN-SIM step only
#          3) Runs perfsuite with 10 TimeSize Events, without running Scimark on the spare cores and passing fake conditions to cmsDriver.py
#  can be defined as:

global listperfsuitekeywords
listperfsuitekeywords = [{                                                                                            }, # empty dictionary = default run
                         {"TimeSizeEvents": 25, "IgProfEvents" :0, "ValgrindEvents":0, "cores": [1,2], "candles": ["MinBias"]                                }, 
                         {"TimeSizeEvents": 10, "IgProfEvents" :0, "ValgrindEvents":0,  "runonspare": False, "cmsdriverOptions": "--conditions=FakeConditions"}]
