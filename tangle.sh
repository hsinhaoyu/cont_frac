#!/bin/bash
# Tangle files with Org mode
# Note that org-babel-tangle-append is a customized command from https://emacs.stackexchange.com/questions/28098/how-to-change-org-mode-babel-tangle-write-to-file-way-as-append-instead-of-overr

/Applications/Emacs.app/Contents/MacOS/Emacs -Q --batch --eval "
    (progn
      (require 'ob-tangle)

    (defun org-babel-tangle-append ()
        (interactive)
        (cl-letf (((symbol-function 'delete-file) #'ignore))
                (org-babel-tangle)))

      (dolist (file command-line-args-left)
        (with-current-buffer (find-file-noselect file)
          (org-babel-tangle-append))))
  " "$@"
