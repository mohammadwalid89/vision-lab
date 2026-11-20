# Task {task}: program the weights of an image classifier head

Work only inside this folder. Do not read or search outside it; do not install packages.
Python with numpy is available.

Data: {dataset}, two classes (label 0 = class {a}, label 1 = class {b}). Each image is 768 numbers:
DINOv2-small CLS token (first 384) and mean patch token (last 384).
Head: score = X @ coef + intercept; score > 0 means class 1. start_head.npz was trained on fit.npz
(only 16 images per class), so it makes mistakes.

Tools: fit.npz and feedback.npz (X, y) to read freely. `python score.py [file]` scores a candidate on
feedback.npz and lists mistakes. Budget: {budget} calls. candidate.npz starts as a copy of start_head.
Rules: write the weights yourself. No sklearn, torch, scipy.optimize, np.linalg.lstsq/solve on labels,
and no gradient steps on the labels. Reading statistics to decide an edit is fine.

The goal is accuracy on a LARGE set of unseen test images from the same two classes, which you never see.
The feedback set is for checking, not for fitting: edits tuned to a few feedback images do not transfer.

Read PLAYBOOK.md (research on how to do this well){lessons_line}

Required method: build at least THREE different candidate heads with different techniques from the playbook
(and METHODS.md if present). Estimate each one's accuracy on unseen data by cross-validation on the labelled
images you have (never by its in-sample feedback score). Pick the best, or combine them if a combination
wins in CV. Record every candidate's CV estimate in NOTES.md. Then update METHODS.md (a running scoreboard
across tasks: technique, task type, CV estimate, and later the true test result) so future tasks learn which
techniques win where.
Finish: best head in final.npz (coef, intercept). Append to NOTES.md: diagnosis, edit, why you expect it
to transfer, weights changed, calls used.
