"""BARC ARC-Heavy as a FlatPool (data/pools/barc.npz): LLM-written generators, re-run and audited in a sandbox.

The programs are untrusted LLM-written code.  They run only in sandboxed worker processes (sandbox.py): macOS
sandbox-exec; elsewhere only with the explicit opt-out BARC_UNSANDBOXED=1 (rlimits and an audit hook, no network
or write barrier).  In order, from the repository root:

    python scripts/fetch_data.py                          # data/raw
    python -m frm.external.corpus                      # data/pools/{rearc,a2_new}.npz: select dedups against them
    python -m frm.external.barc.fetch                  # data/external/barc/: data_100k.jsonl, lib/, venv/
    python -m frm.external.barc.sandbox --selftest     # containment check
    python -m frm.external.barc.build parse
    python -m frm.external.barc.build execute
    python -m frm.external.barc.build filter
    python -m frm.external.barc.build select
    python -m frm.external.barc.audit                  # hours: every selected program x every held-out task
    python -m frm.external.barc.build write            # data/pools/barc{,_val}.{npz,jsonl} + reports
"""
