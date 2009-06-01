#! /usr/bin/env python

from Validation.Tools.GenObject import *
    
if __name__ == "__main__":
    import optparse
    parser = optparse.OptionParser("usage: %prog [options]")
    modeGroup    = optparse.OptionGroup (parser, "Mode Conrols")
    tupleGroup   = optparse.OptionGroup (parser, "Tuple Controls")
    optionsGroup = optparse.OptionGroup (parser, "Options") 
    # mode group
    modeGroup.add_option ('--compare', dest='compare', action='store_true',
                          help='Compare tuple1 to tuple2')
    modeGroup.add_option ('--saveAs', dest='saveAs', type='string',
                          help='Save tuple1 as GO Root file')
    # tuple group
    tupleGroup.add_option ('--tuple1', dest='tuple1', type='string',
                           default='GenObject',
                           help="Tuple type of 1st tuple")
    tupleGroup.add_option ('--tuple2', dest='tuple2', type='string',
                           default='pat',
                           help="Tuple type of 2nd tuple")
    tupleGroup.add_option ('--file1', dest='file1', type='string',
                           default="go1.root",
                           help="1st tuple file")
    tupleGroup.add_option ('--file2', dest='file2', type='string',
                           default="PatAnalyzerSkeletonSkim.root",
                           help="2nd tuple file")
    # options group
    optionsGroup.add_option ('--config', dest='config', type='string',
                             default='config.txt',
                             help="Configuration file (default: '%default')")
    optionsGroup.add_option ('--printEvent', dest='printEvent',
                             action='store_true',
                             help='Prints loaded event to screen')
    optionsGroup.add_option ('--printGlobal', dest='printStatic',
                             action='store_true',
                             help='Prints out global information' +
                             ' (for development)')
    optionsGroup.add_option ('--blur1', dest='blur', type='float',
                             default=0.,
                             help="Randomly changes values by 'BLUR'  " +\
                             "from tuple1.  For debugging only.")
    optionsGroup.add_option ('--blurRate', dest='blurRate', type='float',
                             default=0.02,
                             help="Rate at which objects will be changed. " + \
                             "(%default default)")
    parser.add_option_group (modeGroup)
    parser.add_option_group (tupleGroup)
    parser.add_option_group (optionsGroup)
    (options, args) = parser.parse_args()
    # Here we go
    random.seed( os.getpid() )
    GenObject.loadConfigFile (options.config)
    ROOT.gSystem.Load("libFWCoreFWLite.so")
    ROOT.AutoLibraryLoader.enable()
    # We don't want to use options beyond the main code, so let thee
    # kitchen sink know what we want
    if options.printEvent:
        GenObject._kitchenSinkDict['printEvent'] = True
    if options.blur:
        GenObject._kitchenSinkDict['blur'] = options.blur
    GenObject._kitchenSinkDict['blurRate'] = options.blurRate
    if options.compare:
        # Compare two files
        chain1 = GenObject.prepareTuple (options.tuple1, options.file1)
        chain2 = GenObject.prepareTuple (options.tuple2, options.file2)
        problems = GenObject.compareTwoTrees (chain1, chain2)
        print "problems"
        pprint.pprint (problems)
    if options.saveAs:
        chain1 = GenObject.prepareTuple (options.tuple1, options.file1)
        GenObject.saveTupleAs (chain1, options.saveAs)
    if options.printStatic:
        GenObject.printStatic()

