require "./graph"
require "./executable_graph"

# Compiler: translates a visual GraphData into an ExecutableGraph.
# Phase 1: compiles a single transformer expert (no cooperative ensemble).
# The graph should contain: embedding → transformer blocks → layer_norm → output_head → loss

module ConstructionKit
  class Compiler
    getter vocab_size : Int32
    getter seq_len : Int32

    def initialize(@vocab_size : Int32, @seq_len : Int32 = 128)
    end

    # Compile a pipeline-level graph into an executable graph.
    # For Phase 1, we look for a single expert path:
    #   tokens → embedding → [transformer_block × N] → final_norm → output_head → loss
    def compile(graph : GraphData) : ExecutableGraph
      eg = ExecutableGraph.new

      # Walk the graph nodes and create executable nodes
      graph.nodes.each do |node|
        exec_node = compile_node(node, graph)
        eg.add_node(exec_node) if exec_node
      end

      # Translate graph edges to executable edges
      # Boundary ports (nodeId == -1) map to "__boundary__"
      # Only add edges where both endpoints exist in the executable graph
      graph.edges.each do |edge|
        from_id = edge.from.nodeId == -1 ? "__boundary__" : edge.from.nodeId.to_s
        to_id = edge.to.nodeId == -1 ? "__boundary__" : edge.to.nodeId.to_s

        from_ok = from_id == "__boundary__" || eg.nodes.has_key?(from_id)
        to_ok = to_id == "__boundary__" || eg.nodes.has_key?(to_id)
        next unless from_ok && to_ok

        eg.add_edge(from_id, edge.from.portId, to_id, edge.to.portId)
      end

      eg.compute_topo_order!
      eg
    end

    # Compile a container node by looking at its children graph.
    # For a "transformer" container, compile the internal chain.
    def compile_container(node : Node) : ExecutableGraph?
      children = node.children
      return nil unless children

      case node.type
      when "transformer"
        compile_transformer_internal(node, children)
      else
        nil
      end
    end

    private def compile_node(node : Node, graph : GraphData) : ExecutableNode?
      case node.type
      when "embedding"
        d_model = node.param_i("d_model", 64)
        EmbeddingExec.new(node.id.to_s, @vocab_size, d_model, @seq_len)

      when "attention_layer"
        d_model = node.param_i("d_model", 64)
        n_heads = node.param_i("n_heads", 4)
        d_ff = d_model * 4  # not used by attention, but needed for block
        # Standalone attention — wrap in a config-less block?
        # For Phase 1, attention only appears inside transformer blocks.
        # If we see it at this level, wrap it as a TransformerBlock-less attention.
        nil  # skip — handled inside transformer container

      when "ffn_layer"
        nil  # skip — handled inside transformer container

      when "layer_norm"
        dim = node.param_i("dim", 64)
        LayerNormExec.new(node.id.to_s, dim)

      when "output_head"
        d_model = node.param_i("d_model", 64)
        OutputHeadExec.new(node.id.to_s, d_model, @vocab_size)

      when "loss"
        LossExec.new(node.id.to_s)

      when "transformer"
        # A transformer expert container — compile its internal graph
        # into a chain of executable nodes
        compile_transformer_container(node)

      when "cooperative"
        # Phase 1: if cooperative has one transformer child, compile just that
        compile_cooperative_phase1(node)

      # Data pipeline nodes — not executable, but we need to pass data through
      when "char_tokenizer", "bpe_tokenizer"
        nil  # tokenization is handled by the dataset at the builder level

      when "sequential_window", "sliding_window", "random_window"
        nil  # windowing is handled by the dataset at the builder level

      when "zero_init", "random_init", "learned_init"
        nil  # stream initialization — Phase 2

      when "optimizer"
        nil  # optimizer params are extracted separately

      else
        nil  # unknown node type, skip
      end
    end

    # Compile a transformer container into a flat sequence of executable nodes.
    # Looks at the children graph for the internal chain:
    #   embedding → [LN → Attn → LN → FFN] × N → output_head
    # For Phase 1, we compile each "row" (LN+Attn+LN+FFN) as a single TransformerBlockExec.
    private def compile_transformer_container(node : Node) : ExecutableNode?
      children = node.children
      unless children
        # No children — create from params directly
        d_model = node.param_i("d_model", 64)
        n_layers = node.param_i("n_layers", 3)
        return compile_transformer_from_params(node.id.to_s, d_model, n_layers)
      end

      # Has children graph — compile it
      # For now, extract params and build directly
      # (Full child-graph interpretation is Phase 3)
      d_model = node.param_i("d_model", 64)
      n_layers = node.param_i("n_layers", 3)
      compile_transformer_from_params(node.id.to_s, d_model, n_layers)
    end

    # Build a transformer as a flat chain from params
    private def compile_transformer_from_params(prefix : String, d_model : Int32, n_layers : Int32) : ExecutableNode?
      # For Phase 1, we return a composite node that wraps MiniGPT-like behavior
      # but as a chain: embedding → blocks → final_norm → output_head
      # This is handled by the SingleExpertExec wrapper
      SingleExpertExec.new(prefix, @vocab_size, d_model, n_layers, d_model * 4, @seq_len)
    end

    # Phase 1: cooperative with a single expert path
    private def compile_cooperative_phase1(node : Node) : ExecutableNode?
      children = node.children
      return nil unless children

      # Find transformer experts in children
      transformers = children.nodes.select { |n| n.type == "transformer" }
      return nil if transformers.empty?

      # For Phase 1, just compile the first transformer
      first = transformers.first
      d_model = first.param_i("d_model", 64)
      n_layers = first.param_i("n_layers", 3)
      SingleExpertExec.new(node.id.to_s, @vocab_size, d_model, n_layers, d_model * 4, @seq_len)
    end

    private def compile_transformer_internal(node : Node, graph : GraphData) : ExecutableGraph
      eg = ExecutableGraph.new
      graph.nodes.each do |child|
        exec_node = compile_node(child, graph)
        eg.add_node(exec_node) if exec_node
      end
      graph.edges.each do |edge|
        from_id = edge.from.nodeId == -1 ? "__boundary__" : edge.from.nodeId.to_s
        to_id = edge.to.nodeId == -1 ? "__boundary__" : edge.to.nodeId.to_s
        eg.add_edge(from_id, edge.from.portId, to_id, edge.to.portId)
      end
      eg.compute_topo_order!
      eg
    end
  end

  # ── Single Expert Wrapper ──────────────────────────────────────────────────
  # Wraps a complete single-expert path: embedding → blocks → norm → output
  # This is the Phase 1 workhorse — one node that does what MiniGPT does.
  # Inputs:  { "in" => Array(Int32) (token ids), "targets" => Array(Int32) }
  # Outputs: { "logits" => Mat }

  class SingleExpertExec < ExecutableNode
    getter embedding : MicroGPT::Embedding
    getter blocks : Array(MicroGPT::TransformerBlock)
    getter final_norm : MicroGPT::LayerNorm
    getter output_head : MicroGPT::OutputHead
    getter seq_len : Int32

    @last_logits : MicroGPT::Mat?

    def initialize(id : String, vocab_size : Int32, d_model : Int32, n_layers : Int32, d_ff : Int32, @seq_len : Int32)
      super(id, "single_expert")
      @embedding = MicroGPT::Embedding.new(vocab_size, d_model, @seq_len)
      config = MicroGPT::Config.new
      config.d_model = d_model
      config.n_heads = Math.max(1, d_model // 16)
      config.d_ff = d_ff
      config.seq_len = @seq_len
      @blocks = Array(MicroGPT::TransformerBlock).new(n_layers) { MicroGPT::TransformerBlock.new(config) }
      @final_norm = MicroGPT::LayerNorm.new(d_model)
      @output_head = MicroGPT::OutputHead.new(d_model, vocab_size)
    end

    def forward(inputs : Hash(String, Tensor)) : Hash(String, Tensor)
      ids = inputs["in"].as(Array(Int32))
      x = @embedding.forward(ids)
      @blocks.each { |b| x = b.forward(x) }
      x = @final_norm.forward(x)
      logits = @output_head.forward(x)
      @last_logits = logits
      {"logits" => logits.as(Tensor)}
    end

    def backward(output_grads : Hash(String, MicroGPT::Mat)) : Hash(String, MicroGPT::Mat)
      grad = output_grads["logits"]
      grad = @output_head.proj.backward(grad)
      grad = @final_norm.backward(grad)
      @blocks.reverse_each { |b| grad = b.backward(grad) }
      @embedding.backward(grad)
      {} of String => MicroGPT::Mat
    end

    def update(lr : Float64)
      @embedding.update(lr)
      @blocks.each &.update(lr)
      @final_norm.update(lr)
      @output_head.update(lr)
    end

    def weight_mats : Array(MicroGPT::Mat)
      mats = [] of MicroGPT::Mat
      mats << @embedding.token_emb
      @blocks.each do |b|
        mats << b.attn.wq.w << b.attn.wq.b
        mats << b.attn.wk.w << b.attn.wk.b
        mats << b.attn.wv.w << b.attn.wv.b
        mats << b.attn.wo.w << b.attn.wo.b
        mats << b.ln1.gamma << b.ln1.beta
        mats << b.ff.l1.w << b.ff.l1.b
        mats << b.ff.l2.w << b.ff.l2.b
        mats << b.ln2.gamma << b.ln2.beta
      end
      mats << @final_norm.gamma << @final_norm.beta
      mats << @output_head.proj.w << @output_head.proj.b
      mats
    end

    def adam_mats : Array(MicroGPT::Mat)
      mats = [] of MicroGPT::Mat
      mats << @embedding.adam_tok.m << @embedding.adam_tok.v
      @blocks.each do |b|
        mats << b.attn.wq.adam_w.m << b.attn.wq.adam_w.v
        mats << b.attn.wq.adam_b.m << b.attn.wq.adam_b.v
        mats << b.attn.wk.adam_w.m << b.attn.wk.adam_w.v
        mats << b.attn.wk.adam_b.m << b.attn.wk.adam_b.v
        mats << b.attn.wv.adam_w.m << b.attn.wv.adam_w.v
        mats << b.attn.wv.adam_b.m << b.attn.wv.adam_b.v
        mats << b.attn.wo.adam_w.m << b.attn.wo.adam_w.v
        mats << b.attn.wo.adam_b.m << b.attn.wo.adam_b.v
        mats << b.ln1.adam_gamma.m << b.ln1.adam_gamma.v
        mats << b.ln1.adam_beta.m << b.ln1.adam_beta.v
        mats << b.ff.l1.adam_w.m << b.ff.l1.adam_w.v
        mats << b.ff.l1.adam_b.m << b.ff.l1.adam_b.v
        mats << b.ff.l2.adam_w.m << b.ff.l2.adam_w.v
        mats << b.ff.l2.adam_b.m << b.ff.l2.adam_b.v
        mats << b.ln2.adam_gamma.m << b.ln2.adam_gamma.v
        mats << b.ln2.adam_beta.m << b.ln2.adam_beta.v
      end
      mats << @final_norm.adam_gamma.m << @final_norm.adam_gamma.v
      mats << @final_norm.adam_beta.m << @final_norm.adam_beta.v
      mats << @output_head.proj.adam_w.m << @output_head.proj.adam_w.v
      mats << @output_head.proj.adam_b.m << @output_head.proj.adam_b.v
      mats
    end

    def param_count : Int64
      count = 0_i64
      weight_mats.each { |m| count += m.data.size }
      count
    end
  end
end
