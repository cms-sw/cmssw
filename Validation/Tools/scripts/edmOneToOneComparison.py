#! /usr/bin/env python

if __name__ == "__main__":
    import optparse
    parser = optparse.OptionParser("usage: %prog [options] config.txt file1.root [file2.root]\nVisit https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsToolsEdmOneToOneComparison\nfor full documentation.")
    modeGroup    = optparse.OptionGroup (parser, "Mode Conrols")
    tupleGroup   = optparse.OptionGroup (parser, "Tuple Controls")
    optionsGroup = optparse.OptionGroup (parser, "Options") 
    # mode group
    modeGroup.add_option ('--compare', dest='compare', action='store_true',
                          help='Compare tuple1 to tuple2')
    modeGroup.add_option ('--saveAs', dest='saveAs', type='string',
                          help='Save tuple1 as GO Root file')    
    modeGroup.add_option ('--printTuple', dest='printTuple',
                          action='store_true',
                          help='Print out all events in tuple1')
    modeGroup.add_option ('--interactive', dest='interactive',
                          action='store_true',
                          help='Loads files and prepares "event" '
                          'for interactive mode')
    # tuple group
    tupleGroup.add_option ('--tuple', dest='tuple', type='string',
                           default='',
                           help="Tuple type of 1st and 2nd tuple")
    tupleGroup.add_option ('--tuple1', dest='tuple1', type='string',
                           default='reco',
                           help="Tuple type of 1st tuple")
    tupleGroup.add_option ('--tuple2', dest='tuple2', type='string',
                           default='reco',
                           help="Tuple type of 2nd tuple")
    tupleGroup.add_option ('--file', dest='file', type='string',
                           default="",
                           help="1st and 2nd tuple file (debugging only)")
    tupleGroup.add_option ('--file1', dest='file1', type='string',
                           default="",
                           help="1st tuple file")
    tupleGroup.add_option ('--file2', dest='file2', type='string',
                           default="",
                           help="2nd tuple file")
    tupleGroup.add_option ('--numEvents', dest='numEvents', type='int',
                           default=0,
                           help="number of events for first and second file")
    tupleGroup.add_option ('--numEvents1', dest='numEvents1', type='int',
                           default=0,
                           help="number of events for first file")
    tupleGroup.add_option ('--numEvents2', dest='numEvents2', type='int',
                           default=0,
                           help="number of events for second file")
    tupleGroup.add_option ('--alias', dest='alias', type='string',
                           action='append',
                           help="Change alias ('tuple:object:alias')")
    tupleGroup.add_option ('--label', dest='label', type='string',
                           action='append',
                           help="Change label ('tuple^object^label')")
    tupleGroup.add_option ('--changeVariable', dest='changeVar', type='string',
                           action='append',
                           help="Change variable filling "
                           "('tuple:objName:varName:def')")
    # options group
    optionsGroup.add_option ('--config', dest='config', type='string',
                             default='config.txt',
                             help="Configuration file (default: '%default')")
    optionsGroup.add_option ('--printEvent', dest='printEvent',
                             action='store_true',
                             help='Prints loaded event to screen')
    optionsGroup.add_option ('--printGlobal', dest='printGlobal',
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
    optionsGroup.add_option ('--compRoot', dest='compRoot', type='string',
                             default='',
                             help="Write out root file for file comparisons")
    optionsGroup.add_option ('--debug', dest='debug', action='store_true',
                             help="Print debugging information")
    optionsGroup.add_option ('--strictPairing', dest='strictPairing',
                             action='store_true',
                             help="Objects are paired uniquely by order in collection")
    optionsGroup.add_option ('--relative', dest='relative',
                             action='store_true', default=True,
                             help='Precision is checked against relative difference')
    optionsGroup.add_option ('--absolute', dest='relative',
                             action='store_false',
                             help='Precision is checked against absolute difference')
    optionsGroup.add_option 
    parser.add_option_group (modeGroup)
    parser.add_option_group (tupleGroup)
    parser.add_option_group (optionsGroup)
    (options, args) = parser.parse_args()
    from Validation.Tools.GenObject import *
    ROOT.gROOT.SetBatch()

    lenArgs = len (args)
    if lenArgs >= 1:
        options.config = args[0]
    if lenArgs >= 2:
        options.file1  = args[1]
    if lenArgs == 3:
        options.file2  = args[2]
    if lenArgs > 3:
        raise RuntimeError, "Too many arguments"
    
    # Here we go
    random.seed( os.getpid() )
    GenObject.loadConfigFile (options.config)
    ROOT.gSystem.Load("libFWCoreFWLite.so")
    ROOT.FWLiteEnabler::enable()
    # Let's parse any args
    doubleColonRE = re.compile (r'(.+):(.+):(.+)')
    if options.alias:
        for arg in options.alias:
            aliasMatch = doubleColonRE.match (arg)
            if aliasMatch:
                print "aM", aliasMatch
                GenObject.changeAlias (aliasMatch.group (1),
                                       aliasMatch.group (2),
                                       aliasMatch.group (3))
                continue
            # if we're here, then we have an argument that we don't understand
            raise RuntimeError, "Unknown alias format '%s'" % arg
    tripleColonRE = re.compile (r'(.+):(.+):(.+):(.+)')
    if options.changeVar:
        for arg in options.changeVar:
            changeMatch = tripleColonRE.match (arg)
            if changeMatch:
                GenObject.changeVariable (changeMatch.group (1),
                                          changeMatch.group (2),
                                         changeMatch.group (3),
                                          changeMatch.group (4))
                continue
            # if we're here, then we have an argument that we don't understand
            raise RuntimeError, "Unknown changeVar format '%s'" % arg
    if options.label:
        for label in options.label:            
            pieces = label.split('^')
            if len (pieces) != 3:
                raise RuntimeError, "Can't process label command '%s'" \
                      % options.label
            GenObject.changeLabel (*pieces)
    # We don't want to use options beyond the main code, so let the
    # kitchen sink know what we want
    GenObject.setGlobalFlag ('printEvent',    options.printEvent)
    GenObject.setGlobalFlag ('debug',         options.debug)
    GenObject.setGlobalFlag ('relative',      options.relative)
    GenObject.setGlobalFlag ('strictPairing', options.strictPairing)
    if options.blur:
        GenObject.setGlobalFlag ('blur',      options.blur)
        GenObject.setGlobalFlag ('blurRate',  options.blurRate)
    # take care of any 'double' options now
    if options.tuple:
        options.tuple1 = options.tuple2 = options.tuple
    if options.file:
        options.file1 = options.file2 = options.file
    if options.numEvents:
        options.numEvents1 = options.numEvents2 = options.numEvents
    if options.compare:
        # Compare two files
        chain1 = GenObject.prepareTuple (options.tuple1, options.file1,
                                         options.numEvents1)
        chain2 = GenObject.prepareTuple (options.tuple2, options.file2,
                                         options.numEvents2)
        problems = \
                 GenObject.compareTwoTrees (chain1, chain2,
                                            diffOutputName = options.compRoot)
        print "Summary"
        pprint.pprint (problems)
    if options.saveAs:
        chain1 = GenObject.prepareTuple (options.tuple1, options.file1,
                                         options.numEvents1)
        GenObject.saveTupleAs (chain1, options.saveAs)
    if options.printTuple:
        print "printing tuple"
        GenObject.setGlobalFlag ('printEvent', True)
        chain1 = GenObject.prepareTuple (options.tuple1, options.file1,
                                         options.numEvents1)
        GenObject.printTuple (chain1)
        #GenObject.saveTupleAs (chain1, options.saveAs)
    if options.printGlobal:
        GenObject.printGlobal()
    if options.interactive:
        chain1 = chain2 = 0
        if len (options.file1):
            chain1 = GenObject.prepareTuple (options.tuple1, options.file1,
                                             options.numEvents1)
        if len (options.file2):
            chain2 = GenObject.prepareTuple (options.tuple2, options.file2)
        #############################################
        ## Load and save command line history when ##
        ## running interactively.                  ##
        #############################################
        import os, readline
        import atexit
        historyPath = os.path.expanduser("~/.pyhistory")

        def save_history (historyPath=historyPath):
            import readline
            readline.write_history_file(historyPath)
            if os.path.exists(historyPath):
                readline.read_history_file(historyPath)

        atexit.register(save_history)
        readline.parse_and_bind("set show-all-if-ambiguous on")
        readline.parse_and_bind("tab: complete")
        if os.path.exists (historyPath) :
            readline.read_history_file(historyPath)
            readline.set_history_length(-1)

