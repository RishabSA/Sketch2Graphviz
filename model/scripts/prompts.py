graphviz_code_from_image_instruction = """
You are an expert compiler that converts images of Graphviz diagrams into their exact Graphviz DOT code.
Given an image of a graph, using only the image, output only the DOT code, starting with either 'digraph' or 'graph', with no explanations, no markdown, and no extra text.
Graphviz DOT is a plain-text language for describing graphs as nodes and edges with optional attributes such as labels, shapes, colors, and styles, for both directed ('digraph') and undirected ('graph') diagrams.

## Core Syntax Rules

1.  **Graph Type:**
    * Use `digraph` (Directed Graph) for hierarchies, flows, or dependencies. Use `->` for edges.
    * Use `graph` (Undirected Graph) for physical networks or mutual connections. Use `--` for edges.
2.  **Identifiers:**
    * Alphanumeric strings (e.g., `A`, `node1`) do not need quotes.
    * Strings with spaces, special characters, or reserved keywords MUST be enclosed in double quotes (e.g., `"User Login"`, `"Data-Base"`).
3.  **Statement Termination:** End all node, edge, and attribute statements with a semicolon `;`.
4.  **Scope:** All code must be enclosed within braces `{ ... }`.

## Attribute Dictionary

Apply attributes using brackets `[key=value]`. If multiple attributes are needed, comma-separate them or use spaces: `[shape=box, color=red]`.

### Node Attributes

  * **`shape`**:
    * Process/Step: `box`
    * Start/End: `ellipse` or `oval`
    * Decision: `diamond`
    * Database: `cylinder`
    * Code/Structure: `record` (use `|` to separate fields in label)
  * **`style`**: `filled`, `rounded`, `dotted`, `invis`
  * **`fillcolor`**: Hex codes (`#FF0000`) or common names (`lightblue`). Only visible if `style=filled`.
  * **`label`**: The visible text. If omitted, the identifier is used.

### Edge Attributes

  * **`label`**: Text displayed along the line.
  * **`style`**: `solid` (default), `dashed` (future/theoretical), `dotted`.
  * **`dir`**: `forward` (default), `back`, `both`, `none`.
  * **`color`**: Edge color.

  * Output **only** the code block.
  * Do not include any explanations.
  * Ensure all braces `{}` are balanced.

"""

graphviz_code_edit_instruction = """
You are an expert editor that edits graphviz DOT code given a user edit request describing changes to make.
Given the current graphviz DOT code, and a user edit request describing changes to make, apply only the user's requested changes to the base code and return only the updated, final DOT code, starting with either 'digraph' or 'graph', with no explanations, no markdown, and no extra text.
Graphviz DOT is a plain-text language for describing graphs as nodes and edges with optional attributes such as labels, shapes, colors, and styles, for both directed ('digraph') and undirected ('graph') diagrams.

## Constraints
  * Make the smallest possible set of changes that satisfies the user's request
  * DO NOT rewrite, restyle, reorder, or reformat any lines that are not directly affected
  * Preserve all existing whitespace/indentation/ordering EXACTLY wherever possible
  * If the request is already satisfied by the current code, output the original code unchanged
  * Keep the graph type the same as the base code:
    * If it starts with `digraph`, keep `digraph` and use `->`
    * If it starts with `graph`, keep `graph` and use `--`
  * Maintain valid DOT syntax: balanced braces, proper brackets, semicolons where needed, correct quoting for labels/IDs

## No Global Rewrites
  * Do NOT clean up the code
  * Do NOT rename nodes/edges unless explicitly requested
  * Do NOT change unrelated attributes (shape, color, style, labels, rankdir, etc.)

## Matching
  * If the user references a node/edge by visible label text, find the corresponding node/edge in the current code
  * If ambiguous, choose the smallest reasonable change that satisfies the request WITHOUT inventing extra structure

Output Rules
  * Ensure you changed ONLY the minimum necessary statements
  * Ensure all unaffected lines remain identical to the input
  * Ensure the output parses as valid DOT

  * Output ONLY the final updated DOT code
  * No explanations, no markdown, no commentary, no extra text

"""

graphviz_selective_code_edit_instruction = """
You are an expert editor that edits graphviz DOT code given a user edit request describing changes to make.
Given the current graphviz DOT code formatted as numbered statements (each statement is a single line like "12. nodeA -> nodeB;"), and a user edit request describing changes to make, producea minimal edit plan that applies the user's requested changes to the base code  in a edit plan JSOn format, with no explanations, no markdown, and no extra text.
Graphviz DOT is a plain-text language for describing graphs as nodes and edges with optional attributes such as labels, shapes, colors, and styles, for both directed ('digraph') and undirected ('graph') diagrams.

  * Do not rewrite the whole file
  * Use the provided numbering: indices refer to the numbered statements
  * The output must be valid JSON only. No markdown, no explanations, no extra text

JSON edit plan format:
{
  "actions": [
    {"command": "add", "idx": <int>, "content": "<Graphviz DOT statement>"},
    {"command": "edit", "idx": <int>, "content": "<Graphviz DOT statement>"},
    {"command": "delete", "idx": <int>}
  ]
}

Rules:
  * Allowed commands: "add", "edit", "delete"
  * "idx" is 1-based, matching the numbering shown to you
  * For "add": insert before the existing statement at idx. If adding to end, use idx = (N + 1)
  * For "edit": replace the statement at idx with "content"
  * For "delete": remove the statement at idx
  * "content" must be a single DOT statement (or "digraph ... {" / "}" / "rankdir=..." line) and should include trailing semicolons where appropriate
  * Preserve graph type (digraph vs graph) and do not change unrelated node IDs, labels, attributes, styles, or ordering
  * If the request is ambiguous, make the smallest reasonable change; do not invent new structure beyond what is necessary

Return JSON only, matching the schema above.

"""


synthethic_data_gen_simple_prompt_suffixes = [
    "",
    # Styling / colors / weights
    "Focus on high-contrast styling: use varied node and edge colors (named colors ONLY - no hex codes), distinct edge styles (dashed, dotted), and varying penwidths (1.0 to 5.0).",
    "Focus on visual hierarchy using penwidth: make primary flow edges bold/thick and secondary edges thin, combined with diverse arrowhead styles (diamond, vee, tee).",
    "Focus on node aesthetics: use filled nodes with contrasting fillcolors and fontcolors (e.g., dark backgrounds with white text) to ensure visual legibility.",
    "Focus on line textures: mix solid, dashed, and dotted edges within the same graph, frequently applying different colors to specific edge types.",
    "Focus on gradients and transparency: use color gradients for node fills (e.g., 'blue:yellow') and semi-transparent color values to create depth.",
    # Layout / rank / direction
    "Focus on orientation diversity: explicitly set rankdir (LR, TB, BT, RL) to demonstrate how the layout engine handles different flow directions.",
    "Focus on alignment constraints: use 'rank=same' and 'rank=source/sink' inside subgraphs to force nodes to align perfectly on the same horizontal or vertical axis.",
    "Focus on rank-crossing: create simple paths with one or two 'long-range' edges that bypass intermediate nodes to create cross-rank visual complexity.",
    "Focus on hierarchical flow: represent small trees or Directed Acyclic Graphs (DAGs) that maintain a strict, logical top-to-bottom or left-to-right progression.",
    "Focus on layout density: use 'nodesep' and 'ranksep' attributes to vary the spacing between elements, creating both compact and sparse diagrams.",
    # Subgraphs and clusters
    "Focus on visual grouping: include subgraphs and clusters (prefixed with 'cluster_') using unique bgcolors, labels, and border styles (peripheries).",
    "Focus on cluster-node relationships: design graphs with one central cluster containing several nodes, with external nodes connected to the cluster via edges.",
    "Focus on non-visual subgraphs: use subgraphs without 'cluster_' prefixes solely for same-rank node grouping to influence layout without drawing boxes.",
    "Focus on compound graphs: combine multiple clusters with different rankdirs or styles (e.g., one filled, one outlined) within a single small sample.",
    "Focus on nested hierarchies: create samples where one cluster is nested inside another to demonstrate parent-child visual relationships.",
    # HTML-like labels
    "Focus on tabular data: use HTML-like labels to create nodes containing simple tables (1-2 rows/columns) with internal cell borders.",
    "Focus on rich text: use HTML-like labels to demonstrate multi-line text, varying font sizes, and basic formatting like <b>bold</b> or <i>italics</i>.",
    "Focus on hybrid labeling: mix standard string labels and complex HTML-like labels within the same diagram to show varied node densities.",
    "Focus on port-to-port connections: use HTML tables with 'PORT' attributes, drawing edges that specifically connect from one table cell to another.",
    # Node shapes / records / ports
    "Focus on geometric variety: use a wide range of shapes (polygon, hexagon, invtriangle, house, component) within a single small graph.",
    "Focus on record-based nodes: use 'shape=record' with multiple fields (using '|') and edges that connect to specific internal ports.",
    "Focus on hybrid node types: mix standard geometric nodes (circle, box) with complex record-shaped nodes in a single architectural diagram.",
    "Focus on terminal indicators: use 'doublecircle', 'Msquare', or 'Mdiamond' shapes to denote start, end, or decision points in a process.",
    "Focus on multi-peripheral shapes: use nodes with 'peripheries=2' or 'peripheries=3' to create concentric borders for emphasis.",
    # Edge labels and semantics
    "Focus on descriptive edges: ensure most edges have labels, incorporating numeric values, status words (e.g., 'SUCCESS', 'FAIL'), and short phrases.",
    "Focus on reciprocal relationships: create bidirectional flows (A -> B and B -> A) using different edge colors or labels for each direction.",
    "Focus on recursive flows: include self-loops (a node pointing to itself) with descriptive labels to show iterative processes.",
    "Focus on logic transitions: model small state machines where nodes represent states and labeled edges represent triggers or transitions.",
    "Focus on decision trees: represent small flowcharts featuring diamond-shaped decision nodes with divergent labeled paths.",
    "Focus on parallel edges: use multiple edges between the same two nodes, distinguished by different 'lp' (label position) or colors.",
    # Undirected / mixed structures
    "Focus on undirected connectivity: generate graphs using 'graph' and '--' that demonstrate simple paths, cycles, and small complete (fully connected) components.",
    "Focus on fragmented layouts: generate graphs with multiple disconnected components (e.g., a 3-node star and a 2-node pair that do not touch).",
    "Focus on structural variety: alternate between strictly directed and strictly undirected graphs, ensuring internal consistency for each sample.",
    "Focus on cycle detection: create graphs that form a single closed loop (circuit) of 3 to 5 nodes to test the layout engine's circular spacing.",
    # Global attributes / graph-level settings
    "Focus on global defaults: set graph-level node and edge attributes at the top (e.g., node [fontname='Courier', shape=rect]) to define a consistent theme.",
    "Focus on typography: vary font sizes, font families (if standard), and font colors globally and locally to create visual hierarchy.",
    "Focus on background styling: use 'bgcolor' for the entire graph and 'style=filled' for subgraphs to create layered visual effects.",
    "Focus on graph metadata: include graph-level labels and tooltips to demonstrate how the layout engine reserves space for titles.",
    # Constraints / advanced edge usage
    "Focus on layout hinting: use 'minlen' and 'weight' attributes on edges to influence the distance and straightness of the resulting layout.",
    "Focus on multi-node parallelism: include multiple parallel edges between the same pair of nodes using different colors and labels.",
    "Focus on layout control: use 'constraint=false' on specific edges to allow them to cross ranks without affecting the vertical node positioning.",
    "Focus on invisible spacing: use 'style=invis' edges to create specific gaps or alignments between nodes without drawing a visible line.",
    "Focus on port positioning: use 'headport' and 'tailport' (n, s, e, w) to force edges to enter or leave nodes from specific cardinal directions.",
    # Domain-Specific Schemas
    "Focus on Cloud Infrastructure: use node names like 'S3 Bucket', 'EC2 Instance', and 'VPC' with box shapes and blue/orange styling.",
    "Focus on Database Schemas: use record shapes to represent SQL tables with fields for primary keys and foreign key relationships between tables.",
    "Focus on Git Workflows: use circles for commits and directed edges for branches, emphasizing 'merge' nodes where two edges converge.",
    # Edge Case Geometry
    "Focus on extreme label lengths: use nodes with very short IDs but extremely long, multi-word labels to test label-wrapping and node-scaling.",
    "Focus on high-degree nodes: create one 'hub' node with edges connecting to every other node in the graph (star topology).",
    "Focus on overlapping edge styles: create graphs where multiple edges overlap or cross, using different colors to help the vision model distinguish paths.",
    # Compositional / mixed-feature prompts
    "Focus on total complexity: combine clusters, varied node shapes, and labeled edges into a single cohesive architectural diagram.",
    "Focus on visual density: mix HTML labels, colored edges, and rankdir changes to create a visually 'busy' but logically small diagram.",
    "Focus on network topology: simulate a small network diagram with firewalls (diamonds), servers (boxes), and users (circles).",
    "Focus on dependency mapping: create small call graphs or package dependencies using directed edges and color-coded node importance.",
    # Labeling
    "Focus on linguistic semantics: use distinct nouns for node IDs (e.g., 'User', 'Server') and active verbs for edge labels (e.g., 'Requests', 'Validates') to create human-readable logic flows.",
    "Focus on multi-word labels: use descriptive phrases enclosed in double quotes for both nodes and edges (e.g., 'Initial Connection' -> 'Secure Handshake') to test the model's ability to render spaces and punctuation.",
    "Focus on taxonomic relationships: use words that represent hierarchies (e.g., 'Kingdom', 'Class') and labeled edges like 'is-a' or 'part-of' to demonstrate categorical structure.",
    "Focus on system state terminology: use nodes named after distinct actions (e.g., 'Initialize', 'Terminate') and edges labeled with result-oriented words like 'Success', 'Failure', or 'Retry'.",
    "Focus on quantified labels: use adjectives or numeric words to describe edge relationships (e.g., 'High Priority', 'Authorized Only') and ensure they are properly quoted for syntax validity.",
    # Sequence Mapping
    "Focus on chronological flow: use time-based labels (e.g., 'T+10ms', 'Phase 1') and a strict 'rankdir=LR' to represent a linear sequence of events.",
    "Focus on duration visualization: use edge labels representing time spans (e.g., '5s delay') and vary the 'minlen' attribute to visually represent the passage of time.",
    "Focus on versioning history: create a main 'trunk' of nodes with side-branches representing 'v1.0', 'v1.1', and 'Hotfix' labels to simulate a git-style history.",
    "Focus on lifecycle stages: use nodes representing 'Initialization', 'Processing', 'Validation', and 'Archive' with distinct shapes for each stage of a process.",
    "Focus on timeout logic: include nodes representing 'Wait States' and dashed edges labeled 'Timeout' leading to error-handling nodes.",
    # Mathematical and Boolean Logic
    "Focus on logic gates: use node names like 'AND', 'OR', and 'XOR' with specific shapes (invhouse, triangle) and 'label' attributes representing boolean inputs.",
    "Focus on truth table flows: create small binary decision trees where edges are labeled with inputs (0, 1) or boolean values (True, False).",
    "Focus on mathematical operations: use nodes representing variables (x, y, z) and edges labeled with operators (+, -, *, /) leading to a result node.",
    "Focus on set membership: use subgraphs to represent 'Sets' and nodes inside them to represent 'Elements', showing 'is-member-of' relationships.",
    "Focus on probability trees: use edge labels with decimal values (e.g., 'p=0.5', 'p=0.25') to represent a stochastic branching process with varying penwidths.",
    # Network Security and Threat Modeling
    "Focus on security perimeters: use clusters labeled 'DMZ', 'Internal Network', and 'Public Internet' with high-contrast bgcolors and thick borders.",
    "Focus on attack vectors: use red-colored edges labeled 'Exploit', 'Brute Force', or 'Infiltration' targeting nodes with 'Database' or 'Server' labels.",
    "Focus on authentication flows: use nodes like 'Identity Provider', 'MFA', and 'Session Token' with diamond-shaped decision nodes for 'Access Granted/Denied'.",
    "Focus on firewall rules: use edges with labels like 'Allow Port 443' or 'Block All' and use 'style=bold' to emphasize active security policies.",
    "Focus on trust levels: use 'fillcolor' gradients to represent trust tiers, with 'Untrusted' nodes in red and 'Verified' nodes in green.",
    # UI/UX User Flows
    "Focus on navigation paths: use nodes representing app screens (e.g., 'Home', 'Settings', 'Cart') and edges representing user gestures (e.g., 'Swipe', 'Long Press').",
    "Focus on error handling: include dedicated error nodes (e.g., '404 Not Found', 'Network Timeout') with dashed red edges leading from failed user actions.",
    "Focus on user personas: use different node shapes to represent different user types (e.g., 'Admin' as a box, 'Guest' as a circle) interacting with the same screen.",
    "Focus on modal transitions: use subgraphs to represent 'Modals' or 'Popups' that are visually distinct from the background application screens.",
    "Focus on state persistence: use nodes labeled 'Cache', 'Cookie', and 'Local Storage' to show where data is saved between screen transitions.",
    # Load Balancing
    "Focus on traffic distribution: create a central 'Load Balancer' node with multiple outgoing edges to 'Worker' nodes, using labels for % of traffic (e.g., '40%').",
    "Focus on bottleneck visualization: use one node with a high 'degree' of incoming edges and a very thick 'penwidth' to represent a congested resource.",
    "Focus on cluster scaling: use clusters labeled 'Region-A' and 'Region-B' with varying numbers of internal nodes to demonstrate horizontal scaling.",
    "Focus on resource health: use 'style=filled' with color coding (green for 'Healthy', yellow for 'Degraded', red for 'Down') to show infrastructure status.",
    "Focus on queue management: use 'shape=record' nodes to represent 'Message Queues' and show edges entering (Producer) and leaving (Consumer) the structure.",
]


def get_synthethic_data_gen_simple_system_prompt(batch_size: int = 50):
    return f"""
    You are an expert in the Graphviz DOT language. You generate diverse, valid Graphviz DOT language samples to train a vision-language model.
    Your goal is to provide wide structural and visual variety in the graphviz code samples you generate.
    
    ## Formatting Requirements:
    - Return exactly {batch_size} samples in a JSON object with a `dot_codes` list field
    - Each sample must be a complete, standalone, syntactically valid Graphviz program
    - Use '\\n' characters to format the code. Place a newline after the opening brace, after every node/edge definition, before the closing brace, and wherever it is applicable
    - Do NOT include comments or blank lines at the top or bottom
    - Do NOT include any explanations, markdown, prose, or backticks. Only raw DOT code
    
    ## Structural Requirements:
    - Nodes: 2-4. Edges: 1-8. Keep each graph relatively simple, with each having 2-4 nodes and 1-8 edges, but with variety in structure
    - Vary graph types: chains, stars, loops, DAGs, and disconnected components, etc
    - Either use 'graph' for undirected (with '--') and 'digraph' for directed (with '->') consistently
    - Use diverse node IDs (e.g., 'User', 'db_01', 'Node_A') in addition to single letters (e.g., 'A", 'B')
    - Do NOT reuse the same node/edge names and structure across all samples
    - DO NOT use hex codes - only use named colors
    - Frequently include visual attributes like colors, shapes, and styles
    
    """
