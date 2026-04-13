module MicroGPT
  module AGPT
    # Weighted next-token loss for one trie prefix node.
    #
    # Given logits for a single prefix state and observed outgoing token counts,
    # computes a mean weighted cross-entropy loss and a logits gradient suitable
    # for correctness-first AGPT training.
    class WeightedNextTokenLoss
      def loss_and_backward(logits : MicroGPT::Mat, counts : Hash(Int32, Int32)) : {Float64, MicroGPT::Mat}
        raise "WeightedNextTokenLoss expects a single-row logits matrix" unless logits.rows == 1
        raise "WeightedNextTokenLoss requires at least one observed next token" if counts.empty?

        probs = MicroGPT.backend.softmax_rows(logits)

        # Force a single bulk CPU download of probs to avoid per-element
        # sync_to_cpu calls (each probs[0,j] read would trigger a sync).
        probs_data = probs.data  # triggers one sync, returns CPU array
        vocab_size = logits.cols

        total = counts.values.sum(0)
        loss = 0.0

        counts.each do |token_id, count|
          loss -= count * Math.log(probs_data[token_id] + 1e-10)
        end
        loss /= total

        # Build grad on CPU directly from the downloaded probs
        grad = MicroGPT::Mat.new(1, vocab_size)
        vocab_size.times do |j|
          grad[0, j] = probs_data[j]
        end
        counts.each do |token_id, count|
          grad[0, token_id] -= count.to_f32 / total
        end

        {loss, grad}
      end
    end
  end
end
