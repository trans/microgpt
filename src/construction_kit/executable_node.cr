require "../microgpt"

# Executable graph nodes — thin wrappers around existing Crystal building blocks.
# Each node has named input/output ports and delegates forward/backward/update
# to the wrapped class.

module ConstructionKit
  # Tagged union for data flowing through graph edges
  alias Tensor = MicroGPT::Mat | Array(Int32)

  # Abstract base for all executable graph nodes
  abstract class ExecutableNode
    getter id : String
    getter type : String

    def initialize(@id : String, @type : String)
    end

    # Forward pass: read named inputs, produce named outputs
    abstract def forward(inputs : Hash(String, Tensor)) : Hash(String, Tensor)

    # Backward pass: receive gradients for outputs, return gradients for inputs
    abstract def backward(output_grads : Hash(String, MicroGPT::Mat)) : Hash(String, MicroGPT::Mat)

    # Update trainable parameters
    abstract def update(lr : Float64)

    # Weight access for serialization / checkpoint
    def weight_mats : Array(MicroGPT::Mat)
      [] of MicroGPT::Mat
    end

    def adam_mats : Array(MicroGPT::Mat)
      [] of MicroGPT::Mat
    end

    def param_count : Int64
      0_i64
    end
  end

  # ── Embedding ────────────────────────────────────────────────────────────────
  # Inputs:  { "in" => Array(Int32) }
  # Outputs: { "out" => Mat [seq_len, d_model] }

  class EmbeddingExec < ExecutableNode
    getter inner : MicroGPT::Embedding

    def initialize(id : String, vocab_size : Int32, d_model : Int32, seq_len : Int32 = 128)
      super(id, "embedding")
      @inner = MicroGPT::Embedding.new(vocab_size, d_model, seq_len)
    end

    def forward(inputs : Hash(String, Tensor)) : Hash(String, Tensor)
      ids = inputs["in"].as(Array(Int32))
      out = @inner.forward(ids)
      {"out" => out.as(Tensor)}
    end

    def backward(output_grads : Hash(String, MicroGPT::Mat)) : Hash(String, MicroGPT::Mat)
      @inner.backward(output_grads["out"])
      {} of String => MicroGPT::Mat  # no input gradient (tokens are discrete)
    end

    def update(lr : Float64)
      @inner.update(lr)
    end

    def weight_mats : Array(MicroGPT::Mat)
      [@inner.token_emb]
    end

    def adam_mats : Array(MicroGPT::Mat)
      [@inner.adam_tok.m, @inner.adam_tok.v]
    end

    def param_count : Int64
      @inner.token_emb.data.size.to_i64
    end
  end

  # ── Transformer Block ────────────────────────────────────────────────────────
  # Pre-norm: x + attn(LN(x)), then x + ffn(LN(x))
  # Inputs:  { "in" => Mat }
  # Outputs: { "out" => Mat }

  class TransformerBlockExec < ExecutableNode
    getter inner : MicroGPT::TransformerBlock

    def initialize(id : String, d_model : Int32, n_heads : Int32, d_ff : Int32, seq_len : Int32)
      super(id, "transformer_block")
      config = MicroGPT::Config.new
      config.d_model = d_model
      config.n_heads = n_heads
      config.d_ff = d_ff
      config.seq_len = seq_len
      @inner = MicroGPT::TransformerBlock.new(config)
    end

    def forward(inputs : Hash(String, Tensor)) : Hash(String, Tensor)
      x = inputs["in"].as(MicroGPT::Mat)
      out = @inner.forward(x)
      {"out" => out.as(Tensor)}
    end

    def backward(output_grads : Hash(String, MicroGPT::Mat)) : Hash(String, MicroGPT::Mat)
      grad = @inner.backward(output_grads["out"])
      {"in" => grad}
    end

    def update(lr : Float64)
      @inner.update(lr)
    end

    def weight_mats : Array(MicroGPT::Mat)
      mats = [] of MicroGPT::Mat
      mats << @inner.attn.wq.w << @inner.attn.wq.b
      mats << @inner.attn.wk.w << @inner.attn.wk.b
      mats << @inner.attn.wv.w << @inner.attn.wv.b
      mats << @inner.attn.wo.w << @inner.attn.wo.b
      mats << @inner.ln1.gamma << @inner.ln1.beta
      mats << @inner.ff.l1.w << @inner.ff.l1.b
      mats << @inner.ff.l2.w << @inner.ff.l2.b
      mats << @inner.ln2.gamma << @inner.ln2.beta
      mats
    end

    def adam_mats : Array(MicroGPT::Mat)
      mats = [] of MicroGPT::Mat
      mats << @inner.attn.wq.adam_w.m << @inner.attn.wq.adam_w.v
      mats << @inner.attn.wq.adam_b.m << @inner.attn.wq.adam_b.v
      mats << @inner.attn.wk.adam_w.m << @inner.attn.wk.adam_w.v
      mats << @inner.attn.wk.adam_b.m << @inner.attn.wk.adam_b.v
      mats << @inner.attn.wv.adam_w.m << @inner.attn.wv.adam_w.v
      mats << @inner.attn.wv.adam_b.m << @inner.attn.wv.adam_b.v
      mats << @inner.attn.wo.adam_w.m << @inner.attn.wo.adam_w.v
      mats << @inner.attn.wo.adam_b.m << @inner.attn.wo.adam_b.v
      mats << @inner.ln1.adam_gamma.m << @inner.ln1.adam_gamma.v
      mats << @inner.ln1.adam_beta.m << @inner.ln1.adam_beta.v
      mats << @inner.ff.l1.adam_w.m << @inner.ff.l1.adam_w.v
      mats << @inner.ff.l1.adam_b.m << @inner.ff.l1.adam_b.v
      mats << @inner.ff.l2.adam_w.m << @inner.ff.l2.adam_w.v
      mats << @inner.ff.l2.adam_b.m << @inner.ff.l2.adam_b.v
      mats << @inner.ln2.adam_gamma.m << @inner.ln2.adam_gamma.v
      mats << @inner.ln2.adam_beta.m << @inner.ln2.adam_beta.v
      mats
    end

    def param_count : Int64
      count = 0_i64
      weight_mats.each { |m| count += m.data.size }
      count
    end
  end

  # ── Layer Norm ───────────────────────────────────────────────────────────────
  # Inputs:  { "in" => Mat }
  # Outputs: { "out" => Mat }

  class LayerNormExec < ExecutableNode
    getter inner : MicroGPT::LayerNorm

    def initialize(id : String, dim : Int32)
      super(id, "layer_norm")
      @inner = MicroGPT::LayerNorm.new(dim)
    end

    def forward(inputs : Hash(String, Tensor)) : Hash(String, Tensor)
      x = inputs["in"].as(MicroGPT::Mat)
      {"out" => @inner.forward(x).as(Tensor)}
    end

    def backward(output_grads : Hash(String, MicroGPT::Mat)) : Hash(String, MicroGPT::Mat)
      {"in" => @inner.backward(output_grads["out"])}
    end

    def update(lr : Float64)
      @inner.update(lr)
    end

    def weight_mats : Array(MicroGPT::Mat)
      [@inner.gamma, @inner.beta]
    end

    def adam_mats : Array(MicroGPT::Mat)
      [@inner.adam_gamma.m, @inner.adam_gamma.v, @inner.adam_beta.m, @inner.adam_beta.v]
    end

    def param_count : Int64
      (@inner.gamma.data.size + @inner.beta.data.size).to_i64
    end
  end

  # ── Output Head ──────────────────────────────────────────────────────────────
  # Inputs:  { "in" => Mat }
  # Outputs: { "out" => Mat [seq_len, vocab_size] (logits) }

  class OutputHeadExec < ExecutableNode
    getter inner : MicroGPT::OutputHead

    def initialize(id : String, d_model : Int32, vocab_size : Int32)
      super(id, "output_head")
      @inner = MicroGPT::OutputHead.new(d_model, vocab_size)
    end

    def forward(inputs : Hash(String, Tensor)) : Hash(String, Tensor)
      x = inputs["in"].as(MicroGPT::Mat)
      {"out" => @inner.forward(x).as(Tensor)}
    end

    def backward(output_grads : Hash(String, MicroGPT::Mat)) : Hash(String, MicroGPT::Mat)
      {"in" => @inner.proj.backward(output_grads["out"])}
    end

    def update(lr : Float64)
      @inner.update(lr)
    end

    def weight_mats : Array(MicroGPT::Mat)
      [@inner.proj.w, @inner.proj.b]
    end

    def adam_mats : Array(MicroGPT::Mat)
      [@inner.proj.adam_w.m, @inner.proj.adam_w.v, @inner.proj.adam_b.m, @inner.proj.adam_b.v]
    end

    def param_count : Int64
      (@inner.proj.w.data.size + @inner.proj.b.data.size).to_i64
    end
  end

  # ── Loss (Cross-Entropy) ─────────────────────────────────────────────────────
  # Inputs:  { "logits_in" => Mat, "targets" => Array(Int32) }
  # Outputs: { } (terminal node — produces loss value and initial gradient)
  # The loss node is special: it computes the loss and produces the gradient
  # that kicks off the backward pass.

  class LossExec < ExecutableNode
    @last_logits : MicroGPT::Mat?
    @last_targets : Array(Int32)?
    @last_loss : Float64 = 0.0
    @last_d_logits : MicroGPT::Mat?

    def initialize(id : String)
      super(id, "loss")
    end

    def forward(inputs : Hash(String, Tensor)) : Hash(String, Tensor)
      logits_tensor = inputs["logits_in"]? || inputs["logits"]?
      targets_tensor = inputs["targets"]? || inputs["target_ids"]?
      raise "LossExec: no logits input" unless logits_tensor
      raise "LossExec: no targets input" unless targets_tensor
      logits = logits_tensor.as(MicroGPT::Mat)
      targets = targets_tensor.as(Array(Int32))
      @last_logits = logits
      @last_targets = targets

      seq_len = logits.rows
      vocab_size = logits.cols

      probs = MicroGPT.backend.softmax_rows(logits)

      loss = 0.0
      targets.each_with_index { |t, i| loss -= Math.log(probs[i, t] + 1e-10) }
      loss /= seq_len
      @last_loss = loss

      # Compute gradient of loss w.r.t. logits
      d_logits = MicroGPT::Mat.new(seq_len, vocab_size)
      seq_len.times do |i|
        vocab_size.times { |j| d_logits[i, j] = probs[i, j] }
        d_logits[i, targets[i]] -= 1.0
      end
      d_logits.scale!(1.0 / seq_len)
      @last_d_logits = d_logits

      {} of String => Tensor
    end

    # Loss node's backward returns the gradient for logits
    def backward(output_grads : Hash(String, MicroGPT::Mat)) : Hash(String, MicroGPT::Mat)
      {"logits_in" => @last_d_logits.not_nil!}
    end

    def update(lr : Float64)
      # no params
    end

    def loss : Float64
      @last_loss
    end
  end

  # ── Attention Layer ──────────────────────────────────────────────────────────
  # Inputs:  { "in" => Mat }
  # Outputs: { "out" => Mat }

  class AttentionExec < ExecutableNode
    getter inner : MicroGPT::MultiHeadAttention

    def initialize(id : String, d_model : Int32, n_heads : Int32, seq_len : Int32)
      super(id, "attention_layer")
      @inner = MicroGPT::MultiHeadAttention.new(d_model, n_heads, seq_len)
    end

    def forward(inputs : Hash(String, Tensor)) : Hash(String, Tensor)
      x = inputs["in"].as(MicroGPT::Mat)
      {"out" => @inner.forward(x).as(Tensor)}
    end

    def backward(output_grads : Hash(String, MicroGPT::Mat)) : Hash(String, MicroGPT::Mat)
      {"in" => @inner.backward(output_grads["out"])}
    end

    def update(lr : Float64)
      @inner.update(lr)
    end

    def weight_mats : Array(MicroGPT::Mat)
      [@inner.wq.w, @inner.wq.b, @inner.wk.w, @inner.wk.b,
       @inner.wv.w, @inner.wv.b, @inner.wo.w, @inner.wo.b]
    end

    def adam_mats : Array(MicroGPT::Mat)
      [@inner.wq.adam_w.m, @inner.wq.adam_w.v, @inner.wq.adam_b.m, @inner.wq.adam_b.v,
       @inner.wk.adam_w.m, @inner.wk.adam_w.v, @inner.wk.adam_b.m, @inner.wk.adam_b.v,
       @inner.wv.adam_w.m, @inner.wv.adam_w.v, @inner.wv.adam_b.m, @inner.wv.adam_b.v,
       @inner.wo.adam_w.m, @inner.wo.adam_w.v, @inner.wo.adam_b.m, @inner.wo.adam_b.v]
    end

    def param_count : Int64
      weight_mats.sum { |m| m.data.size.to_i64 }
    end
  end

  # ── Feed-Forward Layer ───────────────────────────────────────────────────────
  # Inputs:  { "in" => Mat }
  # Outputs: { "out" => Mat }

  class FFNExec < ExecutableNode
    getter inner : MicroGPT::FeedForward

    def initialize(id : String, d_model : Int32, d_ff : Int32)
      super(id, "ffn_layer")
      @inner = MicroGPT::FeedForward.new(d_model, d_ff)
    end

    def forward(inputs : Hash(String, Tensor)) : Hash(String, Tensor)
      x = inputs["in"].as(MicroGPT::Mat)
      {"out" => @inner.forward(x).as(Tensor)}
    end

    def backward(output_grads : Hash(String, MicroGPT::Mat)) : Hash(String, MicroGPT::Mat)
      {"in" => @inner.backward(output_grads["out"])}
    end

    def update(lr : Float64)
      @inner.update(lr)
    end

    def weight_mats : Array(MicroGPT::Mat)
      [@inner.l1.w, @inner.l1.b, @inner.l2.w, @inner.l2.b]
    end

    def adam_mats : Array(MicroGPT::Mat)
      [@inner.l1.adam_w.m, @inner.l1.adam_w.v, @inner.l1.adam_b.m, @inner.l1.adam_b.v,
       @inner.l2.adam_w.m, @inner.l2.adam_w.v, @inner.l2.adam_b.m, @inner.l2.adam_b.v]
    end

    def param_count : Int64
      weight_mats.sum { |m| m.data.size.to_i64 }
    end
  end

  # ── Stream Projection ───────────────────────────────────────────────────────
  # Inputs:  { "in" => Mat }
  # Outputs: { "out" => Mat }

  class StreamProjExec < ExecutableNode
    getter inner : MicroGPT::Linear

    def initialize(id : String, d_in : Int32, d_out : Int32)
      super(id, "stream_proj")
      @inner = MicroGPT::Linear.new(d_in, d_out)
    end

    def forward(inputs : Hash(String, Tensor)) : Hash(String, Tensor)
      x = inputs["in"].as(MicroGPT::Mat)
      {"out" => @inner.forward(x).as(Tensor)}
    end

    def backward(output_grads : Hash(String, MicroGPT::Mat)) : Hash(String, MicroGPT::Mat)
      {"in" => @inner.backward(output_grads["out"])}
    end

    def update(lr : Float64)
      @inner.update(lr)
    end

    def weight_mats : Array(MicroGPT::Mat)
      [@inner.w, @inner.b]
    end

    def adam_mats : Array(MicroGPT::Mat)
      [@inner.adam_w.m, @inner.adam_w.v, @inner.adam_b.m, @inner.adam_b.v]
    end

    def param_count : Int64
      (@inner.w.data.size + @inner.b.data.size).to_i64
    end
  end

  # ── Router ──────────────────────────────────────────────────────────────────
  # Inputs:  { "stream_in" => Mat, "logits_in" => Mat (multi-input, summed) }
  # Outputs: { "logits_out" => Mat }
  # Note: For Phase 2, the router receives pre-blended logits via fan-in.
  # Full router support (multiple logits inputs → weighted blend) needs
  # the router to manage the list internally.

  class GlobalRouterExec < ExecutableNode
    getter inner : MicroGPT::GlobalRouter
    @expert_logits : Array(MicroGPT::Mat) = [] of MicroGPT::Mat
    @last_stream : MicroGPT::Mat?

    def initialize(id : String, n_experts : Int32, stream_dim : Int32, epsilon : Float64 = 0.2)
      super(id, "global_router")
      @inner = MicroGPT::GlobalRouter.new(n_experts, stream_dim)
      @inner.epsilon = epsilon
    end

    def forward(inputs : Hash(String, Tensor)) : Hash(String, Tensor)
      @last_stream = inputs["stream_in"]?.as?(MicroGPT::Mat)

      # Collect expert logits — the graph wires multiple logits_in edges
      # which the executor accumulates. For now, router just passes through.
      if logits = inputs["logits_in"]?.as?(MicroGPT::Mat)
        {"logits_out" => logits.as(Tensor)}
      else
        {} of String => Tensor
      end
    end

    def backward(output_grads : Hash(String, MicroGPT::Mat)) : Hash(String, MicroGPT::Mat)
      if grad = output_grads["logits_out"]?
        {"logits_in" => grad}
      else
        {} of String => MicroGPT::Mat
      end
    end

    def update(lr : Float64)
      # Router weights updated during backward
    end

    def weight_mats : Array(MicroGPT::Mat)
      @inner.weight_mats
    end

    def adam_mats : Array(MicroGPT::Mat)
      @inner.adam_mats
    end

    def param_count : Int64
      @inner.param_count
    end
  end
end
