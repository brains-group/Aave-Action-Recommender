# Remaining audit implementation plan

Baseline simulator 9025cd7765a2812bb8c523affff78573981a026e; parent 0f743a77d75258eb0c575f9241c99d0cb0d01a92. Preserve completed refresh and diagnostics. Only parent .gitignore is pre-existing dirty work.

1. Add collateral-enabled state and contract feasibility gates (active/paused/frozen, caps, eMode, isolation and siloed borrowing). Keep supplied balance separate from its collateral eligibility.
2. Add dated evidence and index interfaces with explicit coverage, exact RAY helpers, asset identities and verified missing jEUR mapping; never attach current state to past observations.
3. Add atomic transaction bundles, explicit beneficiary/payer/recipient routing, aToken transfers/receiveAToken and flash-loan accounting with explicit callback evidence; no external-wallet fabrication.
4. Verify version-specific liquidation behavior against downloaded primary contract sources. Preserve unsupported deployment/date coverage as an error rather than silently select rules.
5. Run focused regressions and bounded cache-based checks, integrate into the separate repositories, update audit status and commands. No paid requests, R execution, manuscript edits or full recommendation rerun.
