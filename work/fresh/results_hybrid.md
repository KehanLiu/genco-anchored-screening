# Hybrid / shrunk start and mismatch gate, policy work/fresh/policy_frozen.json

Fresh load scenarios: files work/goc<N>/*_fresh.parquet only.

Truth: a line above 1.001 of rating after full AC. Columns: missed_critical_states = critical states cleared as secure; missed_overloaded_lines = overloaded lines in those states; secure_flagged = secure states sent to AC (and % of secure states); flagged_pct = share of all states sent to AC; genco_start_share_pct = outages where the gate picked the GENCO-anchored start; mean_newton_iters = full Newton iterations from that start (converged runs). gate = prev1 | hyb1 and sgate = prev1 | shyb1 by smaller mis0 (tie or NaN -> prev). Cushions c = [1.0, 1.0005, 1.001].

Two tables per grid. Table A: loading-only warm-start diagnostic (decision: any line > c; mismatch ignored). These are the original tables, unchanged. Table B: complete frozen policy (decision: any line > c OR post-step mismatch > tau of that start; tau transferred unchanged from the old GOC 2000 N-1 calibration, work/fulln1/report.log lines 93-95; never re-fitted). tau (MW) = {'prev': 0.5, 'genco': 10.0, 'hyb': 5.0, 'shyb': 5.0}; the post-step mismatch is the stored mis1_prev / mis1_hyb / mis1_shyb; for gate / sgate the tau of the start chosen per state applies; a non-finite or missing mis1 counts as above tau (referred); added_referrals = secure referrals under the full policy minus loading-only (same c, same start). Audit finding: the original scorer decided on loading only, although policy_frozen.json specifies one step with the chosen start's tau.

## GOC 10000

shrinkage lambda (va, vm): tiny_seed0 (0.4936, 1.4044), tiny_seed1 (0.4094, 1.2314), small_seed0 (0.6059, 1.3874), small_seed1 (0.7635, 1.3996)

### GOC 10000 A: loading-only warm-start diagnostic (decision: any line > c; mismatch ignored)

| checkpoint | start | c | states | critical_states | missed_critical_states | overloaded_lines | missed_overloaded_lines | secure_states | secure_flagged | secure_flagged_pct | flagged_pct | genco_start_share_pct | mean_newton_iters | newton_converged |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny_seed0 | prev1 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 688 | 89.58 | 91.99 |  | 3.11 | 999 |
| tiny_seed0 | prev1 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 71 | 9.24 | 30.23 |  | 3.11 | 999 |
| tiny_seed0 | prev1 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 8 | 1.04 | 23.92 |  | 3.11 | 999 |
| tiny_seed0 | hyb1 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 700 | 91.15 | 93.19 |  | 2.97 | 999 |
| tiny_seed0 | hyb1 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 92 | 11.98 | 32.33 |  | 2.97 | 999 |
| tiny_seed0 | hyb1 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 19 | 2.47 | 25.03 |  | 2.97 | 999 |
| tiny_seed0 | gate | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 691 | 89.97 | 92.29 | 85.99 | 2.90 | 999 |
| tiny_seed0 | gate | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 72 | 9.38 | 30.33 | 85.99 | 2.90 | 999 |
| tiny_seed0 | gate | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 7 | 0.91 | 23.82 | 85.99 | 2.90 | 999 |
| tiny_seed0 | shyb1 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 689 | 89.71 | 92.09 |  | 3.03 | 999 |
| tiny_seed0 | shyb1 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 74 | 9.64 | 30.53 |  | 3.03 | 999 |
| tiny_seed0 | shyb1 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 7 | 0.91 | 23.82 |  | 3.03 | 999 |
| tiny_seed0 | sgate | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 683 | 88.93 | 91.49 | 91.29 | 3.00 | 999 |
| tiny_seed0 | sgate | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 69 | 8.98 | 30.03 | 91.29 | 3.00 | 999 |
| tiny_seed0 | sgate | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 4 | 0.52 | 23.52 | 91.29 | 3.00 | 999 |
| tiny_seed1 | prev1 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 688 | 89.58 | 91.99 |  | 3.11 | 999 |
| tiny_seed1 | prev1 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 71 | 9.24 | 30.23 |  | 3.11 | 999 |
| tiny_seed1 | prev1 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 8 | 1.04 | 23.92 |  | 3.11 | 999 |
| tiny_seed1 | hyb1 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 692 | 90.10 | 92.39 |  | 2.93 | 999 |
| tiny_seed1 | hyb1 | 1.0005 | 999 | 231 | 2 | 423 | 2 | 768 | 91 | 11.85 | 32.03 |  | 2.93 | 999 |
| tiny_seed1 | hyb1 | 1.0010 | 999 | 231 | 2 | 423 | 2 | 768 | 18 | 2.34 | 24.72 |  | 2.93 | 999 |
| tiny_seed1 | gate | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 684 | 89.06 | 91.59 | 88.69 | 2.87 | 999 |
| tiny_seed1 | gate | 1.0005 | 999 | 231 | 1 | 423 | 1 | 768 | 70 | 9.11 | 30.03 | 88.69 | 2.87 | 999 |
| tiny_seed1 | gate | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 9 | 1.17 | 23.92 | 88.69 | 2.87 | 999 |
| tiny_seed1 | shyb1 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 694 | 90.36 | 92.59 |  | 3.02 | 999 |
| tiny_seed1 | shyb1 | 1.0005 | 999 | 231 | 1 | 423 | 1 | 768 | 75 | 9.77 | 30.53 |  | 3.02 | 999 |
| tiny_seed1 | shyb1 | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 5 | 0.65 | 23.52 |  | 3.02 | 999 |
| tiny_seed1 | sgate | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 689 | 89.71 | 92.09 | 93.49 | 2.99 | 999 |
| tiny_seed1 | sgate | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 70 | 9.11 | 30.13 | 93.49 | 2.99 | 999 |
| tiny_seed1 | sgate | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 5 | 0.65 | 23.62 | 93.49 | 2.99 | 999 |
| small_seed0 | prev1 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 688 | 89.58 | 91.99 |  | 3.11 | 999 |
| small_seed0 | prev1 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 71 | 9.24 | 30.23 |  | 3.11 | 999 |
| small_seed0 | prev1 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 8 | 1.04 | 23.92 |  | 3.11 | 999 |
| small_seed0 | hyb1 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 690 | 89.84 | 92.19 |  | 2.89 | 998 |
| small_seed0 | hyb1 | 1.0005 | 999 | 231 | 1 | 423 | 1 | 768 | 93 | 12.11 | 32.33 |  | 2.89 | 998 |
| small_seed0 | hyb1 | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 18 | 2.34 | 24.82 |  | 2.89 | 998 |
| small_seed0 | gate | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 687 | 89.45 | 91.89 | 88.09 | 2.84 | 999 |
| small_seed0 | gate | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 71 | 9.24 | 30.23 | 88.09 | 2.84 | 999 |
| small_seed0 | gate | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 6 | 0.78 | 23.72 | 88.09 | 2.84 | 999 |
| small_seed0 | shyb1 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 682 | 88.80 | 91.39 |  | 2.96 | 999 |
| small_seed0 | shyb1 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 79 | 10.29 | 31.03 |  | 2.96 | 999 |
| small_seed0 | shyb1 | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 5 | 0.65 | 23.52 |  | 2.96 | 999 |
| small_seed0 | sgate | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 680 | 88.54 | 91.19 | 91.09 | 2.92 | 999 |
| small_seed0 | sgate | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 69 | 8.98 | 30.03 | 91.09 | 2.92 | 999 |
| small_seed0 | sgate | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 3 | 0.39 | 23.42 | 91.09 | 2.92 | 999 |
| small_seed1 | prev1 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 688 | 89.58 | 91.99 |  | 3.11 | 999 |
| small_seed1 | prev1 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 71 | 9.24 | 30.23 |  | 3.11 | 999 |
| small_seed1 | prev1 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 8 | 1.04 | 23.92 |  | 3.11 | 999 |
| small_seed1 | hyb1 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 683 | 88.93 | 91.49 |  | 2.85 | 999 |
| small_seed1 | hyb1 | 1.0005 | 999 | 231 | 4 | 423 | 4 | 768 | 80 | 10.42 | 30.73 |  | 2.85 | 999 |
| small_seed1 | hyb1 | 1.0010 | 999 | 231 | 4 | 423 | 4 | 768 | 10 | 1.30 | 23.72 |  | 2.85 | 999 |
| small_seed1 | gate | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 677 | 88.15 | 90.89 | 90.99 | 2.81 | 999 |
| small_seed1 | gate | 1.0005 | 999 | 231 | 1 | 423 | 1 | 768 | 69 | 8.98 | 29.93 | 90.99 | 2.81 | 999 |
| small_seed1 | gate | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 3 | 0.39 | 23.32 | 90.99 | 2.81 | 999 |
| small_seed1 | shyb1 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 682 | 88.80 | 91.39 |  | 2.92 | 999 |
| small_seed1 | shyb1 | 1.0005 | 999 | 231 | 2 | 423 | 2 | 768 | 74 | 9.64 | 30.33 |  | 2.92 | 999 |
| small_seed1 | shyb1 | 1.0010 | 999 | 231 | 3 | 423 | 3 | 768 | 7 | 0.91 | 23.52 |  | 2.92 | 999 |
| small_seed1 | sgate | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 677 | 88.15 | 90.89 | 92.39 | 2.88 | 999 |
| small_seed1 | sgate | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 68 | 8.85 | 29.93 | 92.39 | 2.88 | 999 |
| small_seed1 | sgate | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 3 | 0.39 | 23.32 | 92.39 | 2.88 | 999 |

### GOC 10000 B: complete frozen policy (decision: any line > c OR post-step mismatch > tau of that start; tau transferred unchanged from the old GOC 2000 N-1 calibration, work/fulln1/report.log lines 93-95; never re-fitted)

| checkpoint | start | tau_mw | c | states | critical_states | missed_critical_states | overloaded_lines | missed_overloaded_lines | secure_states | secure_flagged | secure_flagged_pct | flagged_pct | genco_start_share_pct | mean_newton_iters | newton_converged | added_referrals |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny_seed0 | prev1 | prev 0.5 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 733 | 95.44 | 96.50 |  | 3.11 | 999 | +45 |
| tiny_seed0 | prev1 | prev 0.5 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 475 | 61.85 | 70.67 |  | 3.11 | 999 | +404 |
| tiny_seed0 | prev1 | prev 0.5 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 465 | 60.55 | 69.67 |  | 3.11 | 999 | +457 |
| tiny_seed0 | hyb1 | hyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 702 | 91.41 | 93.39 |  | 2.97 | 999 | +2 |
| tiny_seed0 | hyb1 | hyb 5.0 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 121 | 15.76 | 35.24 |  | 2.97 | 999 | +29 |
| tiny_seed0 | hyb1 | hyb 5.0 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 62 | 8.07 | 29.33 |  | 2.97 | 999 | +43 |
| tiny_seed0 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 694 | 90.36 | 92.59 | 85.99 | 2.90 | 999 | +3 |
| tiny_seed0 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 122 | 15.89 | 35.34 | 85.99 | 2.90 | 999 | +50 |
| tiny_seed0 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 66 | 8.59 | 29.73 | 85.99 | 2.90 | 999 | +59 |
| tiny_seed0 | shyb1 | shyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 693 | 90.23 | 92.49 |  | 3.03 | 999 | +4 |
| tiny_seed0 | shyb1 | shyb 5.0 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 98 | 12.76 | 32.93 |  | 3.03 | 999 | +24 |
| tiny_seed0 | shyb1 | shyb 5.0 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 39 | 5.08 | 27.03 |  | 3.03 | 999 | +32 |
| tiny_seed0 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 688 | 89.58 | 91.99 | 91.29 | 3.00 | 999 | +5 |
| tiny_seed0 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 103 | 13.41 | 33.43 | 91.29 | 3.00 | 999 | +34 |
| tiny_seed0 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 45 | 5.86 | 27.63 | 91.29 | 3.00 | 999 | +41 |
| tiny_seed1 | prev1 | prev 0.5 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 733 | 95.44 | 96.50 |  | 3.11 | 999 | +45 |
| tiny_seed1 | prev1 | prev 0.5 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 475 | 61.85 | 70.67 |  | 3.11 | 999 | +404 |
| tiny_seed1 | prev1 | prev 0.5 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 465 | 60.55 | 69.67 |  | 3.11 | 999 | +457 |
| tiny_seed1 | hyb1 | hyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 694 | 90.36 | 92.59 |  | 2.93 | 999 | +2 |
| tiny_seed1 | hyb1 | hyb 5.0 | 1.0005 | 999 | 231 | 1 | 423 | 1 | 768 | 123 | 16.02 | 35.34 |  | 2.93 | 999 | +32 |
| tiny_seed1 | hyb1 | hyb 5.0 | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 66 | 8.59 | 29.63 |  | 2.93 | 999 | +48 |
| tiny_seed1 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 688 | 89.58 | 91.99 | 88.69 | 2.87 | 999 | +4 |
| tiny_seed1 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0005 | 999 | 231 | 1 | 423 | 1 | 768 | 124 | 16.15 | 35.44 | 88.69 | 2.87 | 999 | +54 |
| tiny_seed1 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 72 | 9.38 | 30.23 | 88.69 | 2.87 | 999 | +63 |
| tiny_seed1 | shyb1 | shyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 698 | 90.89 | 92.99 |  | 3.02 | 999 | +4 |
| tiny_seed1 | shyb1 | shyb 5.0 | 1.0005 | 999 | 231 | 1 | 423 | 1 | 768 | 99 | 12.89 | 32.93 |  | 3.02 | 999 | +24 |
| tiny_seed1 | shyb1 | shyb 5.0 | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 39 | 5.08 | 26.93 |  | 3.02 | 999 | +34 |
| tiny_seed1 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 693 | 90.23 | 92.49 | 93.49 | 2.99 | 999 | +4 |
| tiny_seed1 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 102 | 13.28 | 33.33 | 93.49 | 2.99 | 999 | +32 |
| tiny_seed1 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 43 | 5.60 | 27.43 | 93.49 | 2.99 | 999 | +38 |
| small_seed0 | prev1 | prev 0.5 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 733 | 95.44 | 96.50 |  | 3.11 | 999 | +45 |
| small_seed0 | prev1 | prev 0.5 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 475 | 61.85 | 70.67 |  | 3.11 | 999 | +404 |
| small_seed0 | prev1 | prev 0.5 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 465 | 60.55 | 69.67 |  | 3.11 | 999 | +457 |
| small_seed0 | hyb1 | hyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 692 | 90.10 | 92.39 |  | 2.89 | 998 | +2 |
| small_seed0 | hyb1 | hyb 5.0 | 1.0005 | 999 | 231 | 1 | 423 | 1 | 768 | 123 | 16.02 | 35.34 |  | 2.89 | 998 | +30 |
| small_seed0 | hyb1 | hyb 5.0 | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 61 | 7.94 | 29.13 |  | 2.89 | 998 | +43 |
| small_seed0 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 690 | 89.84 | 92.19 | 88.09 | 2.84 | 999 | +3 |
| small_seed0 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 119 | 15.49 | 35.04 | 88.09 | 2.84 | 999 | +48 |
| small_seed0 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 59 | 7.68 | 29.03 | 88.09 | 2.84 | 999 | +53 |
| small_seed0 | shyb1 | shyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 686 | 89.32 | 91.79 |  | 2.96 | 999 | +4 |
| small_seed0 | shyb1 | shyb 5.0 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 103 | 13.41 | 33.43 |  | 2.96 | 999 | +24 |
| small_seed0 | shyb1 | shyb 5.0 | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 39 | 5.08 | 26.93 |  | 2.96 | 999 | +34 |
| small_seed0 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 685 | 89.19 | 91.69 | 91.09 | 2.92 | 999 | +5 |
| small_seed0 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 106 | 13.80 | 33.73 | 91.09 | 2.92 | 999 | +37 |
| small_seed0 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 45 | 5.86 | 27.63 | 91.09 | 2.92 | 999 | +42 |
| small_seed1 | prev1 | prev 0.5 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 733 | 95.44 | 96.50 |  | 3.11 | 999 | +45 |
| small_seed1 | prev1 | prev 0.5 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 475 | 61.85 | 70.67 |  | 3.11 | 999 | +404 |
| small_seed1 | prev1 | prev 0.5 | 1.0010 | 999 | 231 | 0 | 423 | 0 | 768 | 465 | 60.55 | 69.67 |  | 3.11 | 999 | +457 |
| small_seed1 | hyb1 | hyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 687 | 89.45 | 91.89 |  | 2.85 | 999 | +4 |
| small_seed1 | hyb1 | hyb 5.0 | 1.0005 | 999 | 231 | 4 | 423 | 4 | 768 | 98 | 12.76 | 32.53 |  | 2.85 | 999 | +18 |
| small_seed1 | hyb1 | hyb 5.0 | 1.0010 | 999 | 231 | 4 | 423 | 4 | 768 | 36 | 4.69 | 26.33 |  | 2.85 | 999 | +26 |
| small_seed1 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 682 | 88.80 | 91.39 | 90.99 | 2.81 | 999 | +5 |
| small_seed1 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0005 | 999 | 231 | 1 | 423 | 1 | 768 | 104 | 13.54 | 33.43 | 90.99 | 2.81 | 999 | +35 |
| small_seed1 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 42 | 5.47 | 27.23 | 90.99 | 2.81 | 999 | +39 |
| small_seed1 | shyb1 | shyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 685 | 89.19 | 91.69 |  | 2.92 | 999 | +3 |
| small_seed1 | shyb1 | shyb 5.0 | 1.0005 | 999 | 231 | 2 | 423 | 2 | 768 | 96 | 12.50 | 32.53 |  | 2.92 | 999 | +22 |
| small_seed1 | shyb1 | shyb 5.0 | 1.0010 | 999 | 231 | 3 | 423 | 3 | 768 | 36 | 4.69 | 26.43 |  | 2.92 | 999 | +29 |
| small_seed1 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0000 | 999 | 231 | 0 | 423 | 0 | 768 | 680 | 88.54 | 91.19 | 92.39 | 2.88 | 999 | +3 |
| small_seed1 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0005 | 999 | 231 | 0 | 423 | 0 | 768 | 96 | 12.50 | 32.73 | 92.39 | 2.88 | 999 | +28 |
| small_seed1 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0010 | 999 | 231 | 1 | 423 | 1 | 768 | 36 | 4.69 | 26.63 | 92.39 | 2.88 | 999 | +33 |

## GOC 2000

shrinkage lambda (va, vm): tiny_seed0 (1.1058, 1.2655)

### GOC 2000 A: loading-only warm-start diagnostic (decision: any line > c; mismatch ignored)

| checkpoint | start | c | states | critical_states | missed_critical_states | overloaded_lines | missed_overloaded_lines | secure_states | secure_flagged | secure_flagged_pct | flagged_pct | genco_start_share_pct | mean_newton_iters | newton_converged |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny_seed0 | prev1 | 1.0000 | 2496 | 140 | 0 | 322 | 0 | 2356 | 469 | 19.91 | 24.40 |  | 3.08 | 2496 |
| tiny_seed0 | prev1 | 1.0005 | 2496 | 140 | 2 | 322 | 2 | 2356 | 14 | 0.59 | 6.09 |  | 3.08 | 2496 |
| tiny_seed0 | prev1 | 1.0010 | 2496 | 140 | 3 | 322 | 3 | 2356 | 0 | 0.00 | 5.49 |  | 3.08 | 2496 |
| tiny_seed0 | hyb1 | 1.0000 | 2496 | 140 | 0 | 322 | 0 | 2356 | 442 | 18.76 | 23.32 |  | 2.46 | 2496 |
| tiny_seed0 | hyb1 | 1.0005 | 2496 | 140 | 0 | 322 | 0 | 2356 | 16 | 0.68 | 6.25 |  | 2.46 | 2496 |
| tiny_seed0 | hyb1 | 1.0010 | 2496 | 140 | 0 | 322 | 0 | 2356 | 0 | 0.00 | 5.61 |  | 2.46 | 2496 |
| tiny_seed0 | gate | 1.0000 | 2496 | 140 | 0 | 322 | 0 | 2356 | 441 | 18.72 | 23.28 | 98.28 | 2.46 | 2496 |
| tiny_seed0 | gate | 1.0005 | 2496 | 140 | 0 | 322 | 0 | 2356 | 16 | 0.68 | 6.25 | 98.28 | 2.46 | 2496 |
| tiny_seed0 | gate | 1.0010 | 2496 | 140 | 0 | 322 | 0 | 2356 | 0 | 0.00 | 5.61 | 98.28 | 2.46 | 2496 |
| tiny_seed0 | shyb1 | 1.0000 | 2496 | 140 | 0 | 322 | 0 | 2356 | 441 | 18.72 | 23.28 |  | 2.53 | 2496 |
| tiny_seed0 | shyb1 | 1.0005 | 2496 | 140 | 0 | 322 | 0 | 2356 | 16 | 0.68 | 6.25 |  | 2.53 | 2496 |
| tiny_seed0 | shyb1 | 1.0010 | 2496 | 140 | 0 | 322 | 0 | 2356 | 1 | 0.04 | 5.65 |  | 2.53 | 2496 |
| tiny_seed0 | sgate | 1.0000 | 2496 | 140 | 0 | 322 | 0 | 2356 | 440 | 18.68 | 23.24 | 97.88 | 2.53 | 2496 |
| tiny_seed0 | sgate | 1.0005 | 2496 | 140 | 0 | 322 | 0 | 2356 | 16 | 0.68 | 6.25 | 97.88 | 2.53 | 2496 |
| tiny_seed0 | sgate | 1.0010 | 2496 | 140 | 0 | 322 | 0 | 2356 | 0 | 0.00 | 5.61 | 97.88 | 2.53 | 2496 |

### GOC 2000 B: complete frozen policy (decision: any line > c OR post-step mismatch > tau of that start; tau transferred unchanged from the old GOC 2000 N-1 calibration, work/fulln1/report.log lines 93-95; never re-fitted)

| checkpoint | start | tau_mw | c | states | critical_states | missed_critical_states | overloaded_lines | missed_overloaded_lines | secure_states | secure_flagged | secure_flagged_pct | flagged_pct | genco_start_share_pct | mean_newton_iters | newton_converged | added_referrals |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny_seed0 | prev1 | prev 0.5 | 1.0000 | 2496 | 140 | 0 | 322 | 0 | 2356 | 1461 | 62.01 | 64.14 |  | 3.08 | 2496 | +992 |
| tiny_seed0 | prev1 | prev 0.5 | 1.0005 | 2496 | 140 | 0 | 322 | 0 | 2356 | 1277 | 54.20 | 56.77 |  | 3.08 | 2496 | +1263 |
| tiny_seed0 | prev1 | prev 0.5 | 1.0010 | 2496 | 140 | 0 | 322 | 0 | 2356 | 1274 | 54.07 | 56.65 |  | 3.08 | 2496 | +1274 |
| tiny_seed0 | hyb1 | hyb 5.0 | 1.0000 | 2496 | 140 | 0 | 322 | 0 | 2356 | 455 | 19.31 | 23.84 |  | 2.46 | 2496 | +13 |
| tiny_seed0 | hyb1 | hyb 5.0 | 1.0005 | 2496 | 140 | 0 | 322 | 0 | 2356 | 34 | 1.44 | 6.97 |  | 2.46 | 2496 | +18 |
| tiny_seed0 | hyb1 | hyb 5.0 | 1.0010 | 2496 | 140 | 0 | 322 | 0 | 2356 | 18 | 0.76 | 6.33 |  | 2.46 | 2496 | +18 |
| tiny_seed0 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0000 | 2496 | 140 | 0 | 322 | 0 | 2356 | 466 | 19.78 | 24.28 | 98.28 | 2.46 | 2496 | +25 |
| tiny_seed0 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0005 | 2496 | 140 | 0 | 322 | 0 | 2356 | 49 | 2.08 | 7.57 | 98.28 | 2.46 | 2496 | +33 |
| tiny_seed0 | gate | chosen start: prev 0.5 or hyb 5.0 | 1.0010 | 2496 | 140 | 0 | 322 | 0 | 2356 | 33 | 1.40 | 6.93 | 98.28 | 2.46 | 2496 | +33 |
| tiny_seed0 | shyb1 | shyb 5.0 | 1.0000 | 2496 | 140 | 0 | 322 | 0 | 2356 | 454 | 19.27 | 23.80 |  | 2.53 | 2496 | +13 |
| tiny_seed0 | shyb1 | shyb 5.0 | 1.0005 | 2496 | 140 | 0 | 322 | 0 | 2356 | 35 | 1.49 | 7.01 |  | 2.53 | 2496 | +19 |
| tiny_seed0 | shyb1 | shyb 5.0 | 1.0010 | 2496 | 140 | 0 | 322 | 0 | 2356 | 20 | 0.85 | 6.41 |  | 2.53 | 2496 | +19 |
| tiny_seed0 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0000 | 2496 | 140 | 0 | 322 | 0 | 2356 | 469 | 19.91 | 24.40 | 97.88 | 2.53 | 2496 | +29 |
| tiny_seed0 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0005 | 2496 | 140 | 0 | 322 | 0 | 2356 | 54 | 2.29 | 7.77 | 97.88 | 2.53 | 2496 | +38 |
| tiny_seed0 | sgate | chosen start: prev 0.5 or shyb 5.0 | 1.0010 | 2496 | 140 | 0 | 322 | 0 | 2356 | 38 | 1.61 | 7.13 | 97.88 | 2.53 | 2496 | +38 |

## Summary at c = 1.001 (missed critical states / secure states sent to AC in % of secure states)
| grid | checkpoint | start | loading-only missed | loading-only secure % | full-policy missed | full-policy secure % | added secure referrals |
|---:|:---|:---|---:|---:|---:|---:|---:|
| 10000 | tiny_seed0 | prev1 | 0 | 1.04 (8) | 0 | 60.55 (465) | +457 |
| 10000 | tiny_seed0 | hyb1 | 0 | 2.47 (19) | 0 | 8.07 (62) | +43 |
| 10000 | tiny_seed0 | gate | 0 | 0.91 (7) | 0 | 8.59 (66) | +59 |
| 10000 | tiny_seed0 | shyb1 | 0 | 0.91 (7) | 0 | 5.08 (39) | +32 |
| 10000 | tiny_seed0 | sgate | 0 | 0.52 (4) | 0 | 5.86 (45) | +41 |
| 10000 | tiny_seed1 | prev1 | 0 | 1.04 (8) | 0 | 60.55 (465) | +457 |
| 10000 | tiny_seed1 | hyb1 | 2 | 2.34 (18) | 1 | 8.59 (66) | +48 |
| 10000 | tiny_seed1 | gate | 1 | 1.17 (9) | 1 | 9.38 (72) | +63 |
| 10000 | tiny_seed1 | shyb1 | 1 | 0.65 (5) | 1 | 5.08 (39) | +34 |
| 10000 | tiny_seed1 | sgate | 0 | 0.65 (5) | 0 | 5.60 (43) | +38 |
| 10000 | small_seed0 | prev1 | 0 | 1.04 (8) | 0 | 60.55 (465) | +457 |
| 10000 | small_seed0 | hyb1 | 1 | 2.34 (18) | 1 | 7.94 (61) | +43 |
| 10000 | small_seed0 | gate | 0 | 0.78 (6) | 0 | 7.68 (59) | +53 |
| 10000 | small_seed0 | shyb1 | 1 | 0.65 (5) | 1 | 5.08 (39) | +34 |
| 10000 | small_seed0 | sgate | 0 | 0.39 (3) | 0 | 5.86 (45) | +42 |
| 10000 | small_seed1 | prev1 | 0 | 1.04 (8) | 0 | 60.55 (465) | +457 |
| 10000 | small_seed1 | hyb1 | 4 | 1.30 (10) | 4 | 4.69 (36) | +26 |
| 10000 | small_seed1 | gate | 1 | 0.39 (3) | 1 | 5.47 (42) | +39 |
| 10000 | small_seed1 | shyb1 | 3 | 0.91 (7) | 3 | 4.69 (36) | +29 |
| 10000 | small_seed1 | sgate | 1 | 0.39 (3) | 1 | 4.69 (36) | +33 |
| 2000 | tiny_seed0 | prev1 | 3 | 0.00 (0) | 0 | 54.07 (1274) | +1274 |
| 2000 | tiny_seed0 | hyb1 | 0 | 0.00 (0) | 0 | 0.76 (18) | +18 |
| 2000 | tiny_seed0 | gate | 0 | 0.00 (0) | 0 | 1.40 (33) | +33 |
| 2000 | tiny_seed0 | shyb1 | 0 | 0.04 (1) | 0 | 0.85 (20) | +19 |
| 2000 | tiny_seed0 | sgate | 0 | 0.00 (0) | 0 | 1.61 (38) | +38 |


```
Verification against work/fresh/review_checks/batch3_hybrid_tau_reanalysis.csv: 75 rows compared (75 in the re-analysis CSV, 0 not scored in this run), columns ['load_only_missed', 'load_only_secure_flags', 'tau_missed', 'tau_secure_flags', 'secure_states']: 75 match, 0 differ
 grid  checkpoint start      c  tau_missed_rev  tau_missed_ours  tau_secure_flags_rev  tau_secure_flags_ours  load_only_missed_rev  load_only_missed_ours  load_only_secure_flags_rev  load_only_secure_flags_ours ok
10000  tiny_seed0 prev1 1.0000               0                0                   733                    733                     0                      0                         688                          688 ok
10000  tiny_seed0 prev1 1.0005               0                0                   475                    475                     0                      0                          71                           71 ok
10000  tiny_seed0 prev1 1.0010               0                0                   465                    465                     0                      0                           8                            8 ok
10000  tiny_seed0  hyb1 1.0000               0                0                   702                    702                     0                      0                         700                          700 ok
10000  tiny_seed0  hyb1 1.0005               0                0                   121                    121                     0                      0                          92                           92 ok
10000  tiny_seed0  hyb1 1.0010               0                0                    62                     62                     0                      0                          19                           19 ok
10000  tiny_seed0 shyb1 1.0000               0                0                   693                    693                     0                      0                         689                          689 ok
10000  tiny_seed0 shyb1 1.0005               0                0                    98                     98                     0                      0                          74                           74 ok
10000  tiny_seed0 shyb1 1.0010               0                0                    39                     39                     0                      0                           7                            7 ok
10000  tiny_seed0  gate 1.0000               0                0                   694                    694                     0                      0                         691                          691 ok
10000  tiny_seed0  gate 1.0005               0                0                   122                    122                     0                      0                          72                           72 ok
10000  tiny_seed0  gate 1.0010               0                0                    66                     66                     0                      0                           7                            7 ok
10000  tiny_seed0 sgate 1.0000               0                0                   688                    688                     0                      0                         683                          683 ok
10000  tiny_seed0 sgate 1.0005               0                0                   103                    103                     0                      0                          69                           69 ok
10000  tiny_seed0 sgate 1.0010               0                0                    45                     45                     0                      0                           4                            4 ok
10000  tiny_seed1 prev1 1.0000               0                0                   733                    733                     0                      0                         688                          688 ok
10000  tiny_seed1 prev1 1.0005               0                0                   475                    475                     0                      0                          71                           71 ok
10000  tiny_seed1 prev1 1.0010               0                0                   465                    465                     0                      0                           8                            8 ok
10000  tiny_seed1  hyb1 1.0000               0                0                   694                    694                     0                      0                         692                          692 ok
10000  tiny_seed1  hyb1 1.0005               1                1                   123                    123                     2                      2                          91                           91 ok
10000  tiny_seed1  hyb1 1.0010               1                1                    66                     66                     2                      2                          18                           18 ok
10000  tiny_seed1 shyb1 1.0000               0                0                   698                    698                     0                      0                         694                          694 ok
10000  tiny_seed1 shyb1 1.0005               1                1                    99                     99                     1                      1                          75                           75 ok
10000  tiny_seed1 shyb1 1.0010               1                1                    39                     39                     1                      1                           5                            5 ok
10000  tiny_seed1  gate 1.0000               0                0                   688                    688                     0                      0                         684                          684 ok
10000  tiny_seed1  gate 1.0005               1                1                   124                    124                     1                      1                          70                           70 ok
10000  tiny_seed1  gate 1.0010               1                1                    72                     72                     1                      1                           9                            9 ok
10000  tiny_seed1 sgate 1.0000               0                0                   693                    693                     0                      0                         689                          689 ok
10000  tiny_seed1 sgate 1.0005               0                0                   102                    102                     0                      0                          70                           70 ok
10000  tiny_seed1 sgate 1.0010               0                0                    43                     43                     0                      0                           5                            5 ok
10000 small_seed0 prev1 1.0000               0                0                   733                    733                     0                      0                         688                          688 ok
10000 small_seed0 prev1 1.0005               0                0                   475                    475                     0                      0                          71                           71 ok
10000 small_seed0 prev1 1.0010               0                0                   465                    465                     0                      0                           8                            8 ok
10000 small_seed0  hyb1 1.0000               0                0                   692                    692                     0                      0                         690                          690 ok
10000 small_seed0  hyb1 1.0005               1                1                   123                    123                     1                      1                          93                           93 ok
10000 small_seed0  hyb1 1.0010               1                1                    61                     61                     1                      1                          18                           18 ok
10000 small_seed0 shyb1 1.0000               0                0                   686                    686                     0                      0                         682                          682 ok
10000 small_seed0 shyb1 1.0005               0                0                   103                    103                     0                      0                          79                           79 ok
10000 small_seed0 shyb1 1.0010               1                1                    39                     39                     1                      1                           5                            5 ok
10000 small_seed0  gate 1.0000               0                0                   690                    690                     0                      0                         687                          687 ok
10000 small_seed0  gate 1.0005               0                0                   119                    119                     0                      0                          71                           71 ok
10000 small_seed0  gate 1.0010               0                0                    59                     59                     0                      0                           6                            6 ok
10000 small_seed0 sgate 1.0000               0                0                   685                    685                     0                      0                         680                          680 ok
10000 small_seed0 sgate 1.0005               0                0                   106                    106                     0                      0                          69                           69 ok
10000 small_seed0 sgate 1.0010               0                0                    45                     45                     0                      0                           3                            3 ok
10000 small_seed1 prev1 1.0000               0                0                   733                    733                     0                      0                         688                          688 ok
10000 small_seed1 prev1 1.0005               0                0                   475                    475                     0                      0                          71                           71 ok
10000 small_seed1 prev1 1.0010               0                0                   465                    465                     0                      0                           8                            8 ok
10000 small_seed1  hyb1 1.0000               0                0                   687                    687                     0                      0                         683                          683 ok
10000 small_seed1  hyb1 1.0005               4                4                    98                     98                     4                      4                          80                           80 ok
10000 small_seed1  hyb1 1.0010               4                4                    36                     36                     4                      4                          10                           10 ok
10000 small_seed1 shyb1 1.0000               0                0                   685                    685                     0                      0                         682                          682 ok
10000 small_seed1 shyb1 1.0005               2                2                    96                     96                     2                      2                          74                           74 ok
10000 small_seed1 shyb1 1.0010               3                3                    36                     36                     3                      3                           7                            7 ok
10000 small_seed1  gate 1.0000               0                0                   682                    682                     0                      0                         677                          677 ok
10000 small_seed1  gate 1.0005               1                1                   104                    104                     1                      1                          69                           69 ok
10000 small_seed1  gate 1.0010               1                1                    42                     42                     1                      1                           3                            3 ok
10000 small_seed1 sgate 1.0000               0                0                   680                    680                     0                      0                         677                          677 ok
10000 small_seed1 sgate 1.0005               0                0                    96                     96                     0                      0                          68                           68 ok
10000 small_seed1 sgate 1.0010               1                1                    36                     36                     1                      1                           3                            3 ok
 2000  tiny_seed0 prev1 1.0000               0                0                  1461                   1461                     0                      0                         469                          469 ok
 2000  tiny_seed0 prev1 1.0005               0                0                  1277                   1277                     2                      2                          14                           14 ok
 2000  tiny_seed0 prev1 1.0010               0                0                  1274                   1274                     3                      3                           0                            0 ok
 2000  tiny_seed0  hyb1 1.0000               0                0                   455                    455                     0                      0                         442                          442 ok
 2000  tiny_seed0  hyb1 1.0005               0                0                    34                     34                     0                      0                          16                           16 ok
 2000  tiny_seed0  hyb1 1.0010               0                0                    18                     18                     0                      0                           0                            0 ok
 2000  tiny_seed0 shyb1 1.0000               0                0                   454                    454                     0                      0                         441                          441 ok
 2000  tiny_seed0 shyb1 1.0005               0                0                    35                     35                     0                      0                          16                           16 ok
 2000  tiny_seed0 shyb1 1.0010               0                0                    20                     20                     0                      0                           1                            1 ok
 2000  tiny_seed0  gate 1.0000               0                0                   466                    466                     0                      0                         441                          441 ok
 2000  tiny_seed0  gate 1.0005               0                0                    49                     49                     0                      0                          16                           16 ok
 2000  tiny_seed0  gate 1.0010               0                0                    33                     33                     0                      0                           0                            0 ok
 2000  tiny_seed0 sgate 1.0000               0                0                   469                    469                     0                      0                         440                          440 ok
 2000  tiny_seed0 sgate 1.0005               0                0                    54                     54                     0                      0                          16                           16 ok
 2000  tiny_seed0 sgate 1.0010               0                0                    38                     38                     0                      0                           0                            0 ok
```
