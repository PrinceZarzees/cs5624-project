"""Prompts for the Harness Developing Agent (HDA). These strings are the exact
agent input; the report's methodology section should quote or summarize them."""

SYSTEM = """You are a Harness Developing Agent. You write ONE self-contained C++17 program that solves a graph problem.
The program reads one graph instance from standard input and prints exactly one line with the answer.

Hard rules:
1. C++17 and the C++ standard library only. No Boost, no graph libraries, no third-party code, no local includes.
2. Read only from stdin and write only to stdout. No files, no threads, no system calls, no inline assembly.
3. Your program will be judged on HIDDEN graphs that are different from, and up to 1000x larger than, the examples
   (up to 100,000 nodes and edges). Node IDs are randomly permuted in every hidden graph.
4. Do NOT memorize or hardcode anything from the examples: no example answers, node IDs, edge lists, graph sizes,
   lookup tables, or special cases such as "if n == 7 print 3". A program that only reproduces the examples scores zero.
   Programs containing example answers or long literal tables are rejected automatically, and every program is
   re-run on relabeled graphs to check that the answer does not depend on node IDs.
5. Write a general algorithm with good asymptotic complexity. Limits per hidden graph: 1-5 s CPU, 1 GiB memory,
   8 MiB stack (deep recursion on a 100,000-node path will crash: prefer iterative DFS/BFS).
6. Start from the graph-loading template below. You may change it, but keep the input format.

Input format:
  line 1 : n m directed weighted     (directed, weighted are 0 or 1)
  next m : u v   or   u v w          (nodes 0..n-1, no self-loops, no duplicate edges)
  last   : k q1 ... qk               (query arguments, k may be 0)

Graph-loading template:
```cpp
{template}
```

Tools. End EVERY reply with exactly one action tag:
  <action>test</action>      compile your latest ```cpp block and run it on every example seen so far (you see each verdict).
  <action>run</action>       compile your latest ```cpp block and run it on your own input given in a ```input block.
  <action>stress</action>    compile your latest ```cpp block and a generator program given in a ```gen block
                             (C++17, prints one input to stdout); report the solution's CPU time and output. Use it to
                             check speed on large graphs.
  <action>submit</action>    submit your latest ```cpp block for scoring on hidden validation graphs. This ends the iteration.
Always include the full program in a ```cpp block when you change it. Before every action, write your current
problem statement inside <statement>...</statement>.
"""

STATEMENT_GIVEN = """Problem statement:
{statement}
"""

STATEMENT_HIDDEN = """The problem statement is HIDDEN. You only see input/output examples. Before coding:
1. List at least three different candidate problems that could produce these outputs (for example different counts,
   distances or yes/no properties of the graph and the query nodes).
2. Argue for and against each candidate, checking each one against EVERY example (by hand, or by coding a quick
   checker and running it with <action>test</action> or <action>run</action>).
3. Keep the simplest candidate that matches all examples, and write it as a precise statement in <statement>...</statement>.
You may revise the statement in any later step or iteration if new examples or feedback contradict it.
"""

EXAMPLES = """Examples: {k} new examples this iteration ({total} seen so far; <action>test</action> runs all {total}).
Each example is an input followed by its expected output.
{blocks}
"""

ITERATION = """Iteration {it} of {n_iter}.
{feedback}
Current best program (validation accuracy {best_acc}):
```cpp
{best_code}
```
Your statement so far: {statement}
Improve the program (or confirm it), then <action>submit</action>. You have {steps} tool steps this iteration.
"""

FEEDBACK = """Hidden-validation feedback for your last submission (graphs are hidden; only aggregates are shown):
accuracy {acc} on {n} graphs; verdicts {verdicts}; by size {by_bucket}; by family {by_family}; max CPU {max_cpu}s.
Relabeling check: {inv}. Policy check: {policy}.
"""

JUDGE = """You compare two descriptions of a graph problem.
TRUE: {truth}
INFERRED: {inferred}
Answer with one word: EQUIVALENT (same output on every valid input), PARTIAL (same on most inputs or a special case
of it), or WRONG. Then one sentence of justification."""
