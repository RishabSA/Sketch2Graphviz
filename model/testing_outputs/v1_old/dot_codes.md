# Testing Graphviz DOT Codes

Graph 1: digraph G1 { node [shape=ellipse, style=filled, fillcolor=white]; A1 [fillcolor=gold]; B1 [fillcolor=lightblue]; C1 [fillcolor=lightgreen]; A1 -> C1 [color=black, style=solid, weight=2]; C1 ->A1 [color=red, style=dashed, weight=4, penwidth=2]; }

Graph 2: digraph G19 { rankdir=LR; s19_1 -> s19_2; s19_2 -> s19_3; s19_3 -> s19_4; s19_4 -> s19_5; s19_5 -> s19_6; }

Graph 3: graph G {
node [shape=plaintext];
m [label=<<TABLE BORDER="0"><TR><TD><B>Node M</B></TD></TR></TABLE>>];
c [label=<<TABLE BORDER="0"><TR><TD><B>Node C</B></TD></TR></TABLE>>];
d [label=<<TABLE BORDER="0"><TR><TD><B>Node D</B></TD></TR></TABLE>>];
e [label=<<TABLE BORDER="0"><TR><TD><B>Node E</B></TD></TR></TABLE>>];
m -- c;
m -- d;
m -- e;
}

Graph 4: digraph G27 {
A [label=<<TABLE BORDER="0"><TR><TD><B>A</B></TD></TR></TABLE>>];
B [label=<<TABLE BORDER="0"><TR><TD><B>Beta</B></TD></TR></TABLE>>];
C [label=<<TABLE BORDER="0"><TR><TD><B>Gamma</B></TD></TR></TABLE>>];
A -> B; A -> C;
}

Graph 5: graph G19 { s19_center -- s19_l1; s19_center -- s19_l2; s19_center -- s19_l3; s19_center -- s19_l4; }

Graph 6: digraph G26 { node [shape=hex, style=filled, fillcolor=white]; hA [fillcolor=gold]; hB [fillcolor=lightsteelblue]; hC [fillcolor=lightgreen]; hD [fillcolor=lightpink]; hA -> hB [color=black, style=solid, weight=4, penwidth=2]; hB -> hC [color=blue, style=dotted, weight=1]; hC -> hD [color=gray, style=dashed, weight=2]; }
