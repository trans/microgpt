module MicroGPT
  module AGPT
    # Correctness-first AGPT trainer skeleton.
    #
    # MVP strategy:
    # replay each prefix as a standard token sequence, obtain logits for the
    # final prefix position, then apply WeightedNextTokenLoss using the node's
    # outgoing edge counts.
    #
    # This intentionally does NOT yet implement shared-prefix cached execution or
    # aggregated Jacobian reuse. Those belong to a later AGPT phase.
    class Trainer
      getter corpus : TrieCorpus
      getter loss : WeightedNextTokenLoss
      getter example_count : Int32

      @cursor : Int32 = 0
      @example_nodes : Array(TrieNode)

      def initialize(@corpus : TrieCorpus, @loss = WeightedNextTokenLoss.new)
        @example_nodes = [] of TrieNode
        @corpus.each_observed_node do |node|
          @example_nodes << node unless node.depth == 0
        end
        @example_count = @example_nodes.size
      end

      def each_example(& : TrieCorpus::PrefixExample ->)
        @example_nodes.each do |node|
          yield TrieCorpus::PrefixExample.new(@corpus.prefix_for(node), node.next_token_counts_hash)
        end
      end

      def explain_mvp : String
        "Replay-prefix AGPT trainer skeleton: trie examples + weighted next-token loss."
      end

      def train_step(model : MicroGPT::MiniGPT) : {Float64, TrieCorpus::PrefixExample}
        raise "AGPT::Trainer has no non-empty prefix examples" if @example_nodes.empty?

        node = next_node
        prefix = limit_prefix(@corpus.prefix_for(node), model.config.seq_len)

        logits = model.forward(prefix)
        last_logits = final_position_logits(logits)
        counts = node.next_token_counts_hash
        loss_value, last_grad = @loss.loss_and_backward(last_logits, counts)

        grad = scatter_last_row_gradient(last_grad, logits.rows, logits.cols)
        grad = model.output.proj.backward(grad)
        grad = model.final_norm.backward(grad)
        model.blocks.reverse_each { |block| grad = block.backward(grad) }
        model.embedding.backward(grad)

        lr = model.config.learning_rate
        model.embedding.update(lr)
        model.blocks.each &.update(lr)
        model.final_norm.update(lr)
        model.output.update(lr)

        example = TrieCorpus::PrefixExample.new(prefix, counts)
        {loss_value, example}
      end

      private def next_node : TrieNode
        node = @example_nodes[@cursor]
        @cursor = (@cursor + 1) % @example_nodes.size
        node
      end

      private def limit_prefix(prefix_tokens : Array(Int32), seq_len : Int32) : Array(Int32)
        return prefix_tokens if prefix_tokens.size <= seq_len
        prefix_tokens[-seq_len..]
      end

      private def final_position_logits(logits : MicroGPT::Mat) : MicroGPT::Mat
        last_row = logits.rows - 1
        row = MicroGPT::Mat.new(1, logits.cols)
        logits.cols.times do |col|
          row[0, col] = logits[last_row, col]
        end
        row
      end

      private def scatter_last_row_gradient(last_grad : MicroGPT::Mat, rows : Int32, cols : Int32) : MicroGPT::Mat
        grad = MicroGPT::Mat.new(rows, cols)
        last_row = rows - 1
        cols.times do |col|
          grad[last_row, col] = last_grad[0, col]
        end
        grad
      end
    end
  end
end
