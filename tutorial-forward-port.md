---
title: CMS Offline Software
layout: default
related:
 - { name: Project page, link: 'https://github.com/cms-sw/cmssw' }
 - { name: Feedback, link: 'https://github.com/cms-sw/cmssw/issues/new' }
---

## Tutorial: forward porting changes.

One of the greatest advantages of _git_ is that it is much smarter at handling
conflicts between different contribution to the same repository. In particular
it automatically handles the case of two parallel developments, both changing
the same package in orthogonal manner. As smart as it can possibly be, however,
git does not know about the mass of Higgs or how to resolve conflicts where two
people modified the same line in different ways. This latter case is however
automatically detected by github, which will point out that your changes cannot
be merged automatically.

Tutorial will show you how to quickly resolve the latter problem.

### Before you start.

Please make sure you registered to GitHub and that you have provided them
a ssh public key to access your private repository. For more information see
the [FAQ](faq.html).

We will also assume that there is a non mergeable branch called
`tutorial-unmergeable` in the repository of the user `ktf` (i.e. me :-) )

### Create a CMSSW area

First of all lets create an area for the latest / greatest CMSSW available:

    > scram project CMSSW_7_0_X_2013-07-11-1400
    > cd CMSSW_7_0_X_2013-07-11-1400/src
    > cmsenv

### Try to merge a the unmergeable branch

We then try to merge in our working area the unmergeable branch, via:

    > git cms-merge-topic ktf:tutorial-unmergeable
  
this will fail with a message like:

    Auto-merging IOPool/Streamer/src/StreamerInputSource.cc
    Auto-merging FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    CONFLICT (content): Merge conflict in FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    Automatic merge failed; fix conflicts and then commit the result.
    Unable to merge branch tutorial-unmergeable from repository ktf.

which means that the file `FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc`
has changed between the time `ktf:tutorial-unmergeable` was created and the
the release `CMSSW_7_0_X_2013-07-11-1400`.

### Viewing the conflicting changes.

The conflicting changes can be viewed by using the `git diff` command:

    > git diff
    diff --cc FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    index 8a8dbe5,cb21963..0000000
    --- a/FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    +++ b/FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    @@@ -418,6 -419,7 +419,10 @@@ TFWLiteSelectorBasic::setupNewFile(TFil
                 edm::BranchDescription newBD(prod);
                 newBD.updateFriendlyClassName();
                 newReg->copyProduct(newBD);
    ++<<<<<<< HEAD
    ++=======
    +            // Need to call init to get old branch name.
    ++>>>>>>> ktf/tutorial-unmergeable
               }
               prod.init();
             }
    [eulisse@lxbuild168]/build/ge/CMSSW_7_0_X_2013-07

As expected the file which has problems is
`FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc`. Conflicts are marked the
same way they were on CVS, via the `<<<<<<<` and `>>>>>>>` delimiters, and the
`=======`, the version on top (marked with `HEAD`) is the one which is
available in our workarea. The version on the bottom (marked with
`ktf/tutorial-unmergeable`) is the one which belong to the topic branch we are
trying to merge. In this particular case, we simply want the HEAD version (i.e.
we want to remove the comment). Therefore we do it via our preferred editor
(which is of course `vim`, but here I use nedit to make sure people don't write
me asking how to quit.

    > nedit FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    <remove the conflict>

once we have done it, we can do _diff_ again to see the changes:

    > git diff
    diff --cc FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    index 8a8dbe5,cb21963..0000000
    --- a/FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    +++ b/FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc

we can now commit our changes

    git add FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    git commit

This will ask you to provide a comment for the conflict you just solved:

    Merged tutorial-unmergeable from repository ktf

    Conflicts:
            FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    #
    # It looks like you may be committing a merge.
    # If this is not correct, please remove the file
    #       .git/MERGE_HEAD
    # and try again.

In general it's fine to leave it as it is.

Your branch is now updated and the conflicts are solved. You can push your
changes to `my-cmssw` and update your pull request:

    git push my-cmssw HEAD:tutorial-unmergeable

This will tell git to push the current HEAD of the branch you are on (i.e. the
one containing the merge) to the remote `tutorial-unmergeable` branch.

### Rewriting history and cleaning up your changes

What you have read so far is fine and you can stop reading here if the idea of
rewriting history scares you off.

While completely correct the above mentioned procedure has the disadvantage
that an extra commit will show the fact that you had to update you branch to
keep up with an evolving release, possibly modified by someone else. You can
see this by doing `git log`:

    commit 6953bd5e73815508b9cf54bc708e5b2a25bce2cb
    Merge: d3247fe f677f01
    Author: Giulio Eulisse <giulio.eulisse@cern.ch>
    Date:   Thu Jul 11 21:39:27 2013 +0200

        Merged tutorial-unmergeable from repository ktf

        Conflicts:
            FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc

    commit f677f01460b565682f135fe6c9fee6e47b35e4dc
    Author: wmtan <wmtan@fnal.gov>
    Date:   Wed Jul 10 16:02:40 2013 -0500

        Make BranchDescription not mutable

Sometimes this is not desiderable, because it scatters your changes around. It
is however possible to rewrite history, and improve the look it. This is done
via the rebase command:

    > git rebase CMSSW_7_0_X_2013-07-11-1400

where `CMSSW_7_0_X_2013-07-11-1400` is the release you would like to align to.
This will still fail with something like:

    First, rewinding head to replay your work on top of it...
    Applying: Make BranchDescription not mutable
    Using index info to reconstruct a base tree...
    M       FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    M       IOPool/Streamer/src/StreamerInputSource.cc
    Falling back to patching base and 3-way merge...
    Auto-merging IOPool/Streamer/src/StreamerInputSource.cc
    Auto-merging FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    CONFLICT (content): Merge conflict in FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc
    Failed to merge in the changes.
    Patch failed at 0001 Make BranchDescription not mutable
    The copy of the patch that failed is found in:
       /build/ge/CMSSW_7_0_X_2013-07-11-1400/src/.git/rebase-apply/patch

    When you have resolved this problem, run "git rebase --continue".
    If you prefer to skip this patch, run "git rebase --skip" instead.
    To check out the original branch and stop rebasing, run "git rebase --abort".

you can then check the difference using `git diff`, fix them using your
favourite editor and stage them with:

    git add FWCore/TFWLiteSelector/src/TFWLiteSelectorBasic.cc

and then tell git to continue to the next commit:

    git rebase --continue

This will then get rid of the "merge commit" and your history will look much
more linear. You can now push the branch again using `git push`.

### The dangers of rewriting history

While there is nothing inherently bad about rewriting commit history, in
particular if it is for the sake of improving documentation and clarity of the
commit messages, it can cause havok if someone has been working on top of the
now changed head. For this reason it is raccomended that if you find yourself
using `git rebase` you also push your rewritten history to a different branch,
and create a new pull request, so that the rewritten history is easily
identified as such and people who depend on your changes do not have headaches.

### More advanced options

If you have multiple commits you can even rearrange those by using
the `git rebase --interactive` option.

### The above is all great stuff but I need a quick recipe!

So, here it is: you have pushed your changes onto your branch and made a pull
request! Great - but then the integration team tells you that your pull request
no longer merges. This happens if others have made changes at those code lines
also affected by your changes

So here is the recipe to UPDATE your pull request:

Make a new developer area (eg, based on the most recent IB), e.g.:

    cmsrel CMSSW_7_0_X_2013-12-06-0200
    cd CMSSW_7_0_X_2013-12-06-0200/src
    cmsenv

Update to the HEAD of the CMSSW release series, here `CMSSW_7_0_X`:

    git cms-merge-topic CMSSW_7_0_X

Checkout your old branch (from the pull request which does not merge),
 for example:

    git checkout -b <my-development-branch>

Run the merging of the pull request yourself, such as:

    git cms-merge-topic <pull-request-id>

Look for conflicts:

    git diff

Fix them:

    emacs ...
    emacs ...

Commit them back to the old branch:

    git commit -a -m "Fix conflicts." 

Push the branch:

    git push my-cmssw HEAD:<my-development-branch>

(beware: must add "HEAD:" in the above)

This updates your pull request!

Now wait for the "+1" to arrive.
