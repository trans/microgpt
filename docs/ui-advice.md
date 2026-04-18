# Graph / Node Editor Strategy (AST-based Linear Algebra)

## Core Recommendation

Use **Rete.js** as the editor layer, but architect your system as:

```
Editor Graph (Rete) → Compile → AST → Evaluate / Optimize
```

Do **not** treat the editor graph as your canonical representation.

---

## 1. Node Model (Typed Operators)

Define nodes as **semantic operators**, not UI components.

- Inputs: typed (scalar, vector, matrix)
- Outputs: typed
- Operation: pure function or symbolic transform

Example:

```ts
type PortType = "scalar" | "vector" | "matrix"

type NodeDef = {
  name: string
  inputs: Record<string, PortType>
  outputs: Record<string, PortType>
  evaluate?: (inputs) => outputs
}
```

---

## 2. Strict Separation of Concerns

### Editor Graph (Rete)
- Node positions
- Connections
- UI state

### AST (your system)
- Pure computation graph
- No layout or UI concerns

### Pipeline

```ts
EditorGraph → compile() → AST → evaluate()
```

This enables:
- validation
- optimization passes
- serialization stability
- future backend changes

---

## 3. Type System (Enforce Early)

Enforce type compatibility **at connection time**, not during execution.

```ts
function canConnect(outType: PortType, inType: PortType): boolean
```

Examples:
- scalar → scalar ✅
- matrix → matrix ✅
- scalar → matrix ❌ (unless broadcast explicitly supported)

Optional extension:
- track **shape metadata** (e.g., 3x3, NxM)

---

## 4. Use Rete’s Engine (Don’t Rebuild It)

Leverage:
- dependency resolution
- dataflow execution
- node processing

Avoid building a custom scheduler prematurely.

---

## 5. Evaluation Strategy

Support both:

### A) Immediate (Dataflow)
- recompute on change
- good for live preview

### B) Compiled Execution
- build AST
- run optimized evaluation

You will likely want both:
- preview → dataflow
- final compute → compiled

---

## 6. Node Categories

Define clear node types:

- **Constant nodes** (numbers, matrices)
- **Variable/input nodes**
- **Operator nodes** (add, multiply, transpose, etc.)

Make constants first-class citizens.

---

## 7. Interaction & State Discipline

Even with Rete, enforce:

- Do not store derived values in node state
- Do not mix layout with computation
- Keep transient UI state separate from graph data
- Keep port geometry deterministic

---

## 8. Compile Step (Critical Layer)

Your compile step should:

1. Traverse graph
2. Validate:
   - connectivity
   - types
   - acyclicity (if required)
3. Produce normalized AST

Example output shape:

```ts
type ASTNode = {
  op: string
  inputs: ASTNode[]
  meta?: any
}
```

---

## 9. Optimization Opportunities (Later)

Once AST exists, you can add:

- constant folding
- algebraic simplification
- common subexpression elimination
- shape inference
- execution planning (CPU vs GPU)

---

## 10. When to Consider Moving Beyond Rete

Only if you hit:

- very large graphs (100s–1000s nodes)
- heavy compiler-style optimization passes
- GPU / shader-style execution pipelines

Until then, Rete is sufficient.

---

## 11. Strategic Perspective

Treat the system as:

> **A visual front-end for an algebra engine**

- Rete = interface
- Your AST + evaluator = core system

This keeps your architecture extensible and avoids UI-driven constraints on your math model.

---

## Final Recommendation

Start with:

- Rete.js for editor
- Typed node definitions
- Clean compile step → AST

Delay:
- optimization
- performance tuning
- custom rendering engines

Focus first on:
- correctness
- type safety
- clean separation of layers
