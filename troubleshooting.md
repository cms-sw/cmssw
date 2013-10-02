---
title: CMS Offline Software
layout: default
related:
 - { name: Home, link: index.html }
 - { name: Project, link: 'https://github.com/cms-sw/cmssw' }
 - { name: Feedback, link: 'https://github.com/cms-sw/cmssw/issues/new' }
---

# Troubleshooting git

### Sparse checkout does not work.

Apparently some university deployed a non working `git` 1.7.4 client. This
results in sparse checkout misbehavior. Using 1.7.4.1 or later seems to fix the
issue.

### Missing public key.

In case you get the following message:

    Permission denied (publickey).
    fatal: The remote end hung up unexpectedly

you forgot to register your ssh key when you registered to github, you can do
it by going to <https://github.com/settings/ssh> and adding your public key
there.

### My repository and the official CMSSW repository seem to be completely different!

This probably means that you did not reset your fork on the 26th of June, when
the git migration started. All the copies of the repository before that are to
be considered obsolete.

You can check you have the correct fork by looking at your repository page,
`https://github.com/<username>/cmssw`, if it shows that you have  forked from the
repository called `cms-sw/cmssw-old` it means that you need to fork again.

![old-fork](old-fork.png)

You can fix this by deleting your current fork (look under the settings
directory), and then fork again.
